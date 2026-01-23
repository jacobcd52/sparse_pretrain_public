
"""
Binary next-token prediction tasks for circuit pruning.

Each task defines paired (positive, negative) sequences where only one completion
is correct. The task measures whether the model can distinguish between them.

These are simplified dummy tasks based on SimpleStories-style text.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import torch
from transformers import PreTrainedTokenizer


@dataclass
class TaskExample:
    """A single task example with positive and negative sequences."""
    positive_ids: torch.Tensor  # Token IDs for positive example
    negative_ids: torch.Tensor  # Token IDs for negative example
    correct_token: int  # The correct final token
    incorrect_token: int  # The incorrect final token
    eval_position: int = -1  # Position to evaluate logits (-1 means last token of sequence)


class BinaryTask(ABC):
    """Base class for binary next-token prediction tasks."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Task name."""
        pass
    
    @abstractmethod
    def generate_example(self) -> TaskExample:
        """Generate a single (positive, negative) example pair."""
        pass
    
    def generate_batch(self, batch_size: int, max_length: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of task examples.
        
        Args:
            batch_size: Number of examples to generate
            max_length: Maximum sequence length. If 0, uses dynamic padding 
                       (pad to max length within the batch). This is more efficient.
        
        Returns:
            positive_ids: (batch_size, seq_len) - positive sequences (right-padded)
            negative_ids: (batch_size, seq_len) - negative sequences (right-padded)
            correct_tokens: (batch_size,) - correct final tokens
            incorrect_tokens: (batch_size,) - incorrect final tokens
            eval_positions: (batch_size,) - position to evaluate logits for each example
        """
        examples = [self.generate_example() for _ in range(batch_size)]
        
        # Determine padding length: dynamic (max in batch) or fixed
        if max_length <= 0:
            # Dynamic padding: use max length in this batch
            pad_length = max(len(ex.positive_ids) for ex in examples)
        else:
            pad_length = max_length
        
        # Pad sequences (RIGHT padding - model was not trained with left padding)
        pad_id = self.tokenizer.pad_token_id or 0
        
        positive_ids = []
        negative_ids = []
        correct_tokens = []
        incorrect_tokens = []
        eval_positions = []
        
        for ex in examples:
            # Truncate or pad positive
            pos = ex.positive_ids[:pad_length]
            actual_len = len(pos)
            if actual_len < pad_length:
                pos = torch.cat([pos, torch.full((pad_length - actual_len,), pad_id, dtype=torch.long)])
            positive_ids.append(pos)
            
            # Truncate or pad negative
            neg = ex.negative_ids[:pad_length]
            if len(neg) < pad_length:
                neg = torch.cat([neg, torch.full((pad_length - len(neg),), pad_id, dtype=torch.long)])
            negative_ids.append(neg)
            
            correct_tokens.append(ex.correct_token)
            incorrect_tokens.append(ex.incorrect_token)
            
            # Determine evaluation position
            # If example has explicit eval_position, use it (clamped to actual length)
            # Otherwise, use the last real token position (not pad)
            if ex.eval_position >= 0:
                eval_pos = min(ex.eval_position, actual_len - 1)
            else:
                # Default: evaluate at the last real token (position actual_len - 1)
                eval_pos = actual_len - 1
            eval_positions.append(eval_pos)
        
        return (
            torch.stack(positive_ids),
            torch.stack(negative_ids),
            torch.tensor(correct_tokens, dtype=torch.long),
            torch.tensor(incorrect_tokens, dtype=torch.long),
            torch.tensor(eval_positions, dtype=torch.long),
        )


class DummyQuoteTask(BinaryTask):
    """
    Simplified quote-closing task for SimpleStories.
    
    Task: Given a sentence with an opening quote, predict whether to close with
    single quote (') or double quote (").
    
    Positive: Opens with " -> should close with "
    Negative: Same context but incorrect quote type
    """
    
    # Story fragments to use as context
    STORY_TEMPLATES = [
        "The little girl said {quote}Hello",
        "Mom whispered {quote}Be careful",
        "Dad shouted {quote}Watch out",
        "The teacher explained {quote}This is important",
        "Grandma smiled and said {quote}I love you",
        "The boy asked {quote}Can I play",
        "She replied {quote}Of course",
        "He answered {quote}Yes please",
        "The cat seemed to say {quote}Feed me",
        "Everyone cheered {quote}Hooray",
    ]
    
    @property
    def name(self) -> str:
        return "dummy_quote"
    
    def generate_example(self) -> TaskExample:
        template = self.rng.choice(self.STORY_TEMPLATES)
        
        # Randomly choose if this is a double or single quote example
        use_double = self.rng.random() > 0.5
        
        if use_double:
            open_quote = '"'
            correct_close = '"'
            incorrect_close = "'"
        else:
            open_quote = "'"
            correct_close = "'"
            incorrect_close = '"'
        
        # Create the context (everything before the closing quote)
        context = template.format(quote=open_quote)
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        correct_id = self.tokenizer.encode(correct_close, add_special_tokens=False)
        incorrect_id = self.tokenizer.encode(incorrect_close, add_special_tokens=False)
        
        # Handle case where quote might be multiple tokens
        if len(correct_id) > 0:
            correct_token = correct_id[0]
        else:
            correct_token = self.tokenizer.unk_token_id or 0
            
        if len(incorrect_id) > 0:
            incorrect_token = incorrect_id[0]
        else:
            incorrect_token = self.tokenizer.unk_token_id or 0
        
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        # For this task, positive and negative have same context
        # The difference is only in what token should come next
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class DummyArticleTask(BinaryTask):
    """
    Article prediction task (a vs an).
    
    Task: Given context, predict whether "a" or "an" should come next.
    
    Positive: Word starts with consonant -> "a"
    Negative: Same structure but with vowel word -> "an"
    """
    
    # Words starting with consonants (use "a")
    CONSONANT_WORDS = ["cat", "dog", "bird", "tree", "book", "ball", "toy", "cake", "game", "song"]
    
    # Words starting with vowels (use "an")  
    VOWEL_WORDS = ["apple", "elephant", "ice cream", "orange", "umbrella", "egg", "ant", "owl", "elf", "idea"]
    
    TEMPLATES = [
        "The child wanted {article}",
        "She saw {article}",
        "They found {article}",
        "He picked up {article}",
        "There was {article}",
        "I need {article}",
        "We discovered {article}",
        "The story was about {article}",
    ]
    
    @property
    def name(self) -> str:
        return "dummy_article"
    
    def generate_example(self) -> TaskExample:
        template = self.rng.choice(self.TEMPLATES)
        
        # Randomly choose if this needs "a" or "an"
        use_a = self.rng.random() > 0.5
        
        if use_a:
            word = self.rng.choice(self.CONSONANT_WORDS)
            correct_article = " a"
            incorrect_article = " an"
        else:
            word = self.rng.choice(self.VOWEL_WORDS)
            correct_article = " an"
            incorrect_article = " a"
        
        # Create context (template without article)
        context = template.format(article="")
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        correct_id = self.tokenizer.encode(correct_article, add_special_tokens=False)
        incorrect_id = self.tokenizer.encode(incorrect_article, add_special_tokens=False)
        
        # Get first token of article
        if len(correct_id) > 0:
            correct_token = correct_id[0]
        else:
            correct_token = self.tokenizer.unk_token_id or 0
            
        if len(incorrect_id) > 0:
            incorrect_token = incorrect_id[0]
        else:
            incorrect_token = self.tokenizer.unk_token_id or 0
        
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class DummyPronounTask(BinaryTask):
    """
    Pronoun prediction task (he vs she).
    
    Task: Given a name and context, predict the correct pronoun.
    
    Supports train/val split with different templates but same names.
    Templates are designed to be on-distribution for SimpleStories.
    """
    
    # Top 5 names from SimpleStories dataset (frequency in first 10k stories)
    MALE_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter"]  # 5431, 4163, 3882, 1684, 1021
    FEMALE_NAMES = ["Mia", "Kim", "Rita", "Lily", "Maria"]  # 5622, 4106, 2369, 2268, 1902
    
    # Training templates - "when {name} {action}," format (~50% of fillers)
    # Fillers where exp(-CE_loss) > 0.3, interleaved by difficulty for balanced splits
    TRAIN_TEMPLATES = [
        "when {name} ran to the beach,",
        "when {name} went to the park,",
        "when {name} ran to the garden,",
        "when {name} woke up early,",
        "when {name} explored the cave,",
        "when {name} walked to the store,",
        "when {name} swam in the lake,",
        "when {name} sat on the grass,",
        "when {name} rode a bike,",
        "when {name} played in the garden,",
        "when {name} cleaned the room,",
        "when {name} won the game,",
        "when {name} sat by the fire,",
        "when {name} jumped in the water,",
        "when {name} looked at the stars,",
        "when {name} flew a kite,",
        "when {name} fell asleep,",
        "when {name} drew a picture,",
        "when {name} played with toys,",
        "when {name} dug a hole,",
        "when {name} watched the birds,",
        "when {name} fed the cat,",
        "when {name} solved the puzzle,",
    ]
    
    # Validation templates (~25% of fillers) - used during CARBS tuning
    VAL_TEMPLATES = [
        "when {name} went to the forest,",
        "when {name} went to the kitchen,",
        "when {name} went to school,",
        "when {name} stood by the door,",
        "when {name} closed the window,",
        "when {name} opened the door,",
        "when {name} got tired,",
        "when {name} made a card,",
        "when {name} turned on the light,",
        "when {name} built a fort,",
        "when {name} helped dad,",
    ]
    
    # Superval templates (~25% of fillers) - held out for final evaluation
    SUPERVAL_TEMPLATES = [
        "when {name} walked to the lake,",
        "when {name} climbed the tree,",
        "when {name} went outside,",
        "when {name} sailed the boat,",
        "when {name} stood at the window,",
        "when {name} came home,",
        "when {name} lost the game,",
        "when {name} ate lunch,",
        "when {name} ate dinner,",
        "when {name} played with friends,",
        "when {name} painted a flower,",
    ]
    
    # No continuation needed - pronoun follows directly after comma
    CONTINUATIONS = [
        "",  # Empty - pronoun follows directly
    ]
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        """
        Initialize the pronoun task.
        
        Args:
            tokenizer: Tokenizer to use
            seed: Random seed
            split: "train", "val", or "superval" - determines which templates to use
        """
        super().__init__(tokenizer, seed)
        self.split = split
        if split == "train":
            self.templates = self.TRAIN_TEMPLATES
        elif split == "val":
            self.templates = self.VAL_TEMPLATES
        elif split == "superval":
            self.templates = self.SUPERVAL_TEMPLATES
        else:
            raise ValueError(f"split must be 'train', 'val', or 'superval', got {split}")
    
    @property
    def name(self) -> str:
        return f"dummy_pronoun_{self.split}"
    
    def generate_example(self) -> TaskExample:
        template = self.rng.choice(self.templates)
        continuation = self.rng.choice(self.CONTINUATIONS)
        
        # Randomly choose gender
        use_male = self.rng.random() > 0.5
        
        if use_male:
            name = self.rng.choice(self.MALE_NAMES)
            correct_pronoun = " he"
            incorrect_pronoun = " she"
        else:
            name = self.rng.choice(self.FEMALE_NAMES)
            correct_pronoun = " she"
            incorrect_pronoun = " he"
        
        # Create context
        context = template.format(name=name) + continuation
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        correct_id = self.tokenizer.encode(correct_pronoun, add_special_tokens=False)
        incorrect_id = self.tokenizer.encode(incorrect_pronoun, add_special_tokens=False)
        
        if len(correct_id) > 0:
            correct_token = correct_id[0]
        else:
            correct_token = self.tokenizer.unk_token_id or 0
            
        if len(incorrect_id) > 0:
            incorrect_token = incorrect_id[0]
        else:
            incorrect_token = self.tokenizer.unk_token_id or 0
        
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class DummyPronounWrongTask(DummyPronounTask):
    """
    Pronoun prediction task with INVERTED labels (wrong pronoun as target).
    
    Task: Given a name and context, predict the WRONG pronoun.
    E.g., "when Rita went to the woods," → target is "he" (wrong!)
    
    This is useful for testing what circuits the model uses to NOT predict something.
    """
    
    @property
    def name(self) -> str:
        return f"dummy_pronoun_wrong_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Get the normal example
        example = super().generate_example()
        
        # Swap correct and incorrect tokens
        return TaskExample(
            positive_ids=example.positive_ids,
            negative_ids=example.negative_ids,
            correct_token=example.incorrect_token,  # Swapped!
            incorrect_token=example.correct_token,   # Swapped!
        )


class DummyPronounConstantTokenTask(DummyPronounTask):
    """
    Base class for tasks where target is always a constant token (control tasks).
    
    Uses same contexts as pronoun task but target is always a fixed token.
    This is useful as a control to see what circuits are needed for a constant prediction.
    """
    
    TARGET_TOKEN_STR = " when"  # Override in subclasses
    TASK_SUFFIX = "when"  # Override in subclasses
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed, split)
        # Pre-compute the target token ID
        target_ids = self.tokenizer.encode(self.TARGET_TOKEN_STR, add_special_tokens=False)
        self.target_token = target_ids[0] if target_ids else self.tokenizer.unk_token_id or 0
        # Use "he" as the incorrect token (arbitrary choice)
        he_ids = self.tokenizer.encode(" he", add_special_tokens=False)
        self.he_token = he_ids[0] if he_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"dummy_pronoun_{self.TASK_SUFFIX}_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Get the normal example (just for the context)
        example = super().generate_example()
        
        # Replace target with the constant token
        return TaskExample(
            positive_ids=example.positive_ids,
            negative_ids=example.negative_ids,
            correct_token=self.target_token,
            incorrect_token=self.he_token,
        )


class DummyPronounWhenTask(DummyPronounConstantTokenTask):
    """Task where target is always 'when'."""
    TARGET_TOKEN_STR = " when"
    TASK_SUFFIX = "when"


class DummyPronounIsTask(DummyPronounConstantTokenTask):
    """Task where target is always 'is'."""
    TARGET_TOKEN_STR = " is"
    TASK_SUFFIX = "is"


class DummyPronounEvilTask(DummyPronounConstantTokenTask):
    """Task where target is always 'evil'."""
    TARGET_TOKEN_STR = " evil"
    TASK_SUFFIX = "evil"


class DummyPronounWaterTask(DummyPronounConstantTokenTask):
    """Task where target is always 'water'."""
    TARGET_TOKEN_STR = " water"
    TASK_SUFFIX = "water"


class DummyPronounIsWhenTask(DummyPronounTask):
    """
    Task where target depends on name gender:
    - Male names (Leo, Alex, Samuel, Jose, Peter) → target is "is", incorrect is "when"
    - Female names (Mia, Kim, Rita, Lily, Maria) → target is "when", incorrect is "is"
    
    This tests if a circuit can learn an arbitrary gender→token mapping.
    """
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed, split)
        # Pre-compute token IDs
        is_ids = self.tokenizer.encode(" is", add_special_tokens=False)
        self.is_token = is_ids[0] if is_ids else self.tokenizer.unk_token_id or 0
        when_ids = self.tokenizer.encode(" when", add_special_tokens=False)
        self.when_token = when_ids[0] if when_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"dummy_pronoun_iswhen_{self.split}"
    
    def generate_example(self) -> TaskExample:
        template = self.rng.choice(self.templates)
        continuation = self.rng.choice(self.CONTINUATIONS)
        
        # Randomly choose gender
        use_male = self.rng.random() > 0.5
        
        if use_male:
            name = self.rng.choice(self.MALE_NAMES)
            correct_token = self.is_token    # Male → "is"
            incorrect_token = self.when_token  # incorrect is "when"
        else:
            name = self.rng.choice(self.FEMALE_NAMES)
            correct_token = self.when_token  # Female → "when"
            incorrect_token = self.is_token    # incorrect is "is"
        
        # Create context
        context = template.format(name=name) + continuation
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class DummyTenseTask(BinaryTask):
    """
    Tense prediction task (present vs past).
    
    Task: Given context in present or past tense, predict verb in matching tense.
    
    Example:
    - "When Mia went to the woods, she" → "saw" (past context → past verb)
    - "When Mia goes to the woods, she" → "sees" (present context → present verb)
    
    The task tests whether the model understands tense agreement:
    - Present tense context ("goes", "runs") should be followed by present verb ("sees", "plays")
    - Past tense context ("went", "ran") should be followed by past verb ("saw", "played")
    
    Dataset is built by filtering template+verb combinations where model assigns
    probability > 0.3 to the correct completion.
    """
    
    # All names from pronoun task (no male/female distinction needed for this task)
    ALL_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter", "Mia", "Kim", "Rita", "Lily", "Maria"]
    
    # Name to pronoun mapping (for grammatically correct sentences)
    NAME_TO_PRONOUN = {
        "Leo": "he", "Alex": "he", "Samuel": "he", "Jose": "he", "Peter": "he",
        "Mia": "she", "Kim": "she", "Rita": "she", "Lily": "she", "Maria": "she",
    }
    
    # Verb pairs: (present_3sg, past) - these are completion verbs
    # Prioritized based on what the model actually predicts in story contexts
    VERB_PAIRS = [
        # High probability verbs (from model analysis)
        ("feels", "felt"),
        ("sees", "saw"),
        ("finds", "found"),
        ("notices", "noticed"),
        ("hears", "heard"),
        ("smiles", "smiled"),
        ("thinks", "thought"),
        ("realizes", "realized"),
        ("looks", "looked"),
        ("knows", "knew"),
        ("remembers", "remembered"),
        # Additional common story verbs
        ("laughs", "laughed"),
        ("cries", "cried"),
        ("runs", "ran"),
        ("walks", "walked"),
        ("plays", "played"),
        ("jumps", "jumped"),
        ("stops", "stopped"),
        ("starts", "started"),
        ("decides", "decided"),
        ("wants", "wanted"),
        ("likes", "liked"),
        ("loves", "loved"),
    ]
    
    # Context verb pairs: (present, past) - these set up the tense in the context
    # Template format: "When {name} {ctx_verb} to the {place}, {pron}"
    CONTEXT_VERB_PAIRS = [
        ("goes", "went"),
        ("runs", "ran"),
        ("walks", "walked"),
        ("comes", "came"),
        ("gets", "got"),
    ]
    
    # Template structures: (template_with_present_ctx, template_with_past_ctx)
    # {name} and {pron} are placeholders for name and pronoun
    # The "As {name} sits/sat by the fire" pattern works best based on model analysis
    TEMPLATE_STRUCTURES = [
        # "As" templates (work best with feeling/thinking verbs)
        ("As {name} sits by the fire, {pron}", "As {name} sat by the fire, {pron}"),
        ("As {name} sits by the window, {pron}", "As {name} sat by the window, {pron}"),
        ("As {name} sits on the bench, {pron}", "As {name} sat on the bench, {pron}"),
        ("As {name} sits in the garden, {pron}", "As {name} sat in the garden, {pron}"),
        ("As {name} sits alone, {pron}", "As {name} sat alone, {pron}"),
        ("As {name} walks home, {pron}", "As {name} walked home, {pron}"),
        ("As {name} walks in the park, {pron}", "As {name} walked in the park, {pron}"),
        ("As {name} walks to school, {pron}", "As {name} walked to school, {pron}"),
        ("As {name} plays in the garden, {pron}", "As {name} played in the garden, {pron}"),
        ("As {name} reads a book, {pron}", "As {name} read a book, {pron}"),
        ("As {name} looks at the sky, {pron}", "As {name} looked at the sky, {pron}"),
        ("As {name} looks around, {pron}", "As {name} looked around, {pron}"),
        # "When" templates
        ("When {name} goes to the woods, {pron}", "When {name} went to the woods, {pron}"),
        ("When {name} goes to the park, {pron}", "When {name} went to the park, {pron}"),
        ("When {name} goes home, {pron}", "When {name} went home, {pron}"),
        ("When {name} runs home, {pron}", "When {name} ran home, {pron}"),
        ("When {name} comes home, {pron}", "When {name} came home, {pron}"),
        ("When {name} walks home, {pron}", "When {name} walked home, {pron}"),
        ("When {name} gets home, {pron}", "When {name} got home, {pron}"),
        # "While" templates
        ("While {name} plays outside, {pron}", "While {name} played outside, {pron}"),
        ("While {name} sits quietly, {pron}", "While {name} sat quietly, {pron}"),
        ("While {name} waits for the bus, {pron}", "While {name} waited for the bus, {pron}"),
        ("While {name} thinks about it, {pron}", "While {name} thought about it, {pron}"),
    ]
    
    # Pre-filtered examples where model assigns prob >= 0.1 to correct completion
    # Format: (context_template, present_verb, past_verb, is_present_tense)
    # is_present_tense indicates which tense the context is in
    # 
    # This task uses BINARY cross-entropy loss: softmax over [correct, incorrect]
    # instead of full vocabulary. This allows the task to work even when 
    # individual verb probabilities are low in absolute terms.
    
    # Training templates (~50% of filtered templates)
    TRAIN_TEMPLATES = [
        ("As {name} sat by the window, {pron}", "feels", "felt", False),
        ("As {name} sits on the bench, {pron}", "feels", "felt", True),
        ("As {name} sat on the bench, {pron}", "notices", "noticed", False),
        ("As {name} sat in the garden, {pron}", "feels", "felt", False),
        ("As {name} sat in the garden, {pron}", "notices", "noticed", False),
        ("As {name} walked home, {pron}", "feels", "felt", False),
        ("As {name} walked home, {pron}", "sees", "saw", False),
        ("As {name} walks in the park, {pron}", "sees", "saw", True),
        ("As {name} walked in the park, {pron}", "sees", "saw", False),
        ("As {name} walked in the park, {pron}", "notices", "noticed", False),
        ("As {name} walked to school, {pron}", "sees", "saw", False),
        ("As {name} read a book, {pron}", "feels", "felt", False),
        ("As {name} looked at the sky, {pron}", "feels", "felt", False),
        ("As {name} looks at the sky, {pron}", "sees", "saw", True),
        ("As {name} looked at the sky, {pron}", "sees", "saw", False),
        ("As {name} looked around, {pron}", "notices", "noticed", False),
        ("When {name} went to the woods, {pron}", "finds", "found", False),
        ("When {name} went to the park, {pron}", "feels", "felt", False),
        ("When {name} came home, {pron}", "feels", "felt", False),
        ("When {name} walked home, {pron}", "sees", "saw", False),
        ("When {name} got home, {pron}", "feels", "felt", False),
        ("While {name} played outside, {pron}", "notices", "noticed", False),
        ("While {name} sat quietly, {pron}", "sees", "saw", False),
        ("While {name} sat quietly, {pron}", "notices", "noticed", False),
    ]
    
    # Validation templates (~25% of filtered templates)
    VAL_TEMPLATES = [
        ("As {name} sits by the fire, {pron}", "feels", "felt", True),
        ("As {name} sat by the fire, {pron}", "feels", "felt", False),
        ("As {name} sits by the window, {pron}", "feels", "felt", True),
        ("As {name} sat on the bench, {pron}", "feels", "felt", False),
        ("As {name} sat alone, {pron}", "feels", "felt", False),
        ("As {name} played in the garden, {pron}", "feels", "felt", False),
        ("When {name} went to the woods, {pron}", "feels", "felt", False),
        ("When {name} went to the woods, {pron}", "sees", "saw", False),
        ("When {name} went to the park, {pron}", "sees", "saw", False),
        ("When {name} went home, {pron}", "feels", "felt", False),
        ("When {name} ran home, {pron}", "feels", "felt", False),
        ("While {name} sat quietly, {pron}", "feels", "felt", False),
    ]
    
    # Held-out evaluation templates (~25% of filtered templates)
    SUPERVAL_TEMPLATES = [
        ("As {name} sits in the garden, {pron}", "feels", "felt", True),
        ("As {name} sits alone, {pron}", "feels", "felt", True),
        ("As {name} walked in the park, {pron}", "feels", "felt", False),
        ("As {name} walked to school, {pron}", "feels", "felt", False),
        ("As {name} walked to school, {pron}", "notices", "noticed", False),
        ("As {name} played in the garden, {pron}", "sees", "saw", False),
        ("As {name} played in the garden, {pron}", "notices", "noticed", False),
        ("As {name} looked around, {pron}", "sees", "saw", False),
        ("When {name} walked home, {pron}", "feels", "felt", False),
        ("While {name} played outside, {pron}", "feels", "felt", False),
        ("While {name} played outside, {pron}", "sees", "saw", False),
        ("While {name} waited for the bus, {pron}", "feels", "felt", False),
    ]
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        """
        Initialize the tense task.
        
        Args:
            tokenizer: Tokenizer to use
            seed: Random seed
            split: "train", "val", or "superval" - determines which templates to use
        """
        super().__init__(tokenizer, seed)
        self.split = split
        if split == "train":
            self.templates = self.TRAIN_TEMPLATES
        elif split == "val":
            self.templates = self.VAL_TEMPLATES
        elif split == "superval":
            self.templates = self.SUPERVAL_TEMPLATES
        else:
            raise ValueError(f"split must be 'train', 'val', or 'superval', got {split}")
    
    @property
    def name(self) -> str:
        return f"dummy_tense_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Pick a random template
        template_tuple = self.rng.choice(self.templates)
        context_template, present_verb, past_verb, is_present_tense = template_tuple
        
        # Pick a random name
        name = self.rng.choice(self.ALL_NAMES)
        pronoun = self.NAME_TO_PRONOUN[name]
        
        # Create context
        context = context_template.format(name=name, pron=pronoun)
        
        # Determine correct and incorrect verbs based on context tense
        if is_present_tense:
            correct_verb = " " + present_verb
            incorrect_verb = " " + past_verb
        else:
            correct_verb = " " + past_verb
            incorrect_verb = " " + present_verb
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        correct_id = self.tokenizer.encode(correct_verb, add_special_tokens=False)
        incorrect_id = self.tokenizer.encode(incorrect_verb, add_special_tokens=False)
        
        if len(correct_id) > 0:
            correct_token = correct_id[0]
        else:
            correct_token = self.tokenizer.unk_token_id or 0
            
        if len(incorrect_id) > 0:
            incorrect_token = incorrect_id[0]
        else:
            incorrect_token = self.tokenizer.unk_token_id or 0
        
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )
    
    @classmethod
    def get_all_candidate_examples(cls):
        """
        Generate all possible (template, verb_pair) combinations for filtering.
        
        Returns list of (context_template, present_verb, past_verb, is_present_tense)
        """
        candidates = []
        for present_template, past_template in cls.TEMPLATE_STRUCTURES:
            for present_verb, past_verb in cls.VERB_PAIRS:
                # Present tense context
                candidates.append((present_template, present_verb, past_verb, True))
                # Past tense context  
                candidates.append((past_template, present_verb, past_verb, False))
        return candidates


class IOIStrictTask(BinaryTask):
    """
    Indirect Object Identification (IOI) task.
    
    Task: First sentence introduces one name. Second sentence has a different name
    performing an action that requires a pronoun referring to the first name.
    
    Example: "when Leo looked lost, Mia reminded him"
    - First name: Leo (male)
    - Second name: Mia (acts on Leo)
    - Target: "him" (pronoun referring to Leo)
    
    The pronoun must match the gender of the first name, not the second.
    Uses 5 male and 5 female names (10 total).
    
    Templates tested on ss_bridges model to ensure avg P(correct) >= 50%.
    Total: 587 templates across 8 verbs, ~29,350 unique examples.
    """
    
    MALE_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter"]
    FEMALE_NAMES = ["Mia", "Kim", "Rita", "Lily", "Maria"]
    
    # Templates tested to have P(correct pronoun) >= 0.5 on ss_bridges
    # Format: "when {name1} <situation>, {name2} <verb>" → him/her
    # 8 verbs: welcomed (109), beckoned (108), reminded (108), taught (106),
    #          handed (61), warned (50), told (35), gave (10)
    TEMPLATES = [
        # ==================== WELCOMED (109 templates, 52-83%) ====================
        "when {name1} looked lost, {name2} welcomed",  # 83%
        "when {name1} looked afraid, {name2} welcomed",  # 83%
        "when {name1} looked scared, {name2} welcomed",  # 83%
        "when {name1} looked bored, {name2} welcomed",  # 82%
        "when {name1} felt bored, {name2} welcomed",  # 82%
        "when {name1} walked in, {name2} welcomed",  # 82%
        "when {name1} stumbled, {name2} welcomed",  # 82%
        "when {name1} felt lost, {name2} welcomed",  # 82%
        "when {name1} was scared, {name2} welcomed",  # 82%
        "when {name1} looked worried, {name2} welcomed",  # 82%
        "when {name1} walked inside, {name2} welcomed",  # 81%
        "when {name1} called out, {name2} welcomed",  # 81%
        "when {name1} looked alone, {name2} welcomed",  # 81%
        "when {name1} was afraid, {name2} welcomed",  # 81%
        "when {name1} wondered, {name2} welcomed",  # 81%
        "when {name1} was lost, {name2} welcomed",  # 81%
        "when {name1} looked hungry, {name2} welcomed",  # 81%
        "when {name1} was bored, {name2} welcomed",  # 80%
        "when {name1} looked sick, {name2} welcomed",  # 80%
        "when {name1} looked cold, {name2} welcomed",  # 80%
        "when {name1} was interested, {name2} welcomed",  # 80%
        "when {name1} looked sad, {name2} welcomed",  # 80%
        "when {name1} looked tired, {name2} welcomed",  # 80%
        "when {name1} looked angry, {name2} welcomed",  # 80%
        "when {name1} was sad, {name2} welcomed",  # 80%
        "when {name1} stood alone, {name2} welcomed",  # 79%
        "when {name1} felt scared, {name2} welcomed",  # 79%
        "when {name1} felt afraid, {name2} welcomed",  # 79%
        "when {name1} entered, {name2} welcomed",  # 79%
        "when {name1} looked nervous, {name2} welcomed",  # 79%
        "when {name1} stood there, {name2} welcomed",  # 79%
        "when {name1} was angry, {name2} welcomed",  # 79%
        "when {name1} was hungry, {name2} welcomed",  # 79%
        "when {name1} arrived, {name2} welcomed",  # 79%
        "when {name1} seemed bored, {name2} welcomed",  # 79%
        "when {name1} felt alone, {name2} welcomed",  # 79%
        "when {name1} called, {name2} welcomed",  # 79%
        "when {name1} looked confused, {name2} welcomed",  # 78%
        "when {name1} appeared, {name2} welcomed",  # 78%
        "when {name1} looked hurt, {name2} welcomed",  # 78%
        "when {name1} sat alone, {name2} welcomed",  # 78%
        "when {name1} felt cold, {name2} welcomed",  # 78%
        "when {name1} felt worried, {name2} welcomed",  # 78%
        "when {name1} got scared, {name2} welcomed",  # 78%
        "when {name1} seemed lost, {name2} welcomed",  # 78%
        "when {name1} was worried, {name2} welcomed",  # 78%
        "when {name1} was alone, {name2} welcomed",  # 78%
        "when {name1} seemed afraid, {name2} welcomed",  # 78%
        "when {name1} waited, {name2} welcomed",  # 78%
        "when {name1} felt tired, {name2} welcomed",  # 77%
        "when {name1} came over, {name2} welcomed",  # 77%
        "when {name1} felt angry, {name2} welcomed",  # 77%
        "when {name1} stepped in, {name2} welcomed",  # 77%
        "when {name1} sobbed, {name2} welcomed",  # 77%
        "when {name1} cried, {name2} welcomed",  # 77%
        "when {name1} came in, {name2} welcomed",  # 77%
        "when {name1} felt nervous, {name2} welcomed",  # 77%
        "when {name1} was sick, {name2} welcomed",  # 77%
        "when {name1} stepped inside, {name2} welcomed",  # 77%
        "when {name1} was nervous, {name2} welcomed",  # 77%
        "when {name1} was tired, {name2} welcomed",  # 77%
        "when {name1} slipped, {name2} welcomed",  # 77%
        "when {name1} looked upset, {name2} welcomed",  # 76%
        "when {name1} felt hungry, {name2} welcomed",  # 76%
        "when {name1} seemed angry, {name2} welcomed",  # 76%
        "when {name1} seemed scared, {name2} welcomed",  # 76%
        "when {name1} was hurt, {name2} welcomed",  # 76%
        "when {name1} fell, {name2} welcomed",  # 76%
        "when {name1} hesitated, {name2} welcomed",  # 76%
        "when {name1} asked, {name2} welcomed",  # 76%
        "when {name1} seemed worried, {name2} welcomed",  # 76%
        "when {name1} was curious, {name2} welcomed",  # 76%
        "when {name1} was confused, {name2} welcomed",  # 76%
        "when {name1} felt curious, {name2} welcomed",  # 76%
        "when {name1} was stuck, {name2} welcomed",  # 75%
        "when {name1} felt sad, {name2} welcomed",  # 75%
        "when {name1} felt confused, {name2} welcomed",  # 75%
        "when {name1} was cold, {name2} welcomed",  # 75%
        "when {name1} seemed alone, {name2} welcomed",  # 75%
        "when {name1} felt sick, {name2} welcomed",  # 75%
        "when {name1} got lost, {name2} welcomed",  # 75%
        "when {name1} paused, {name2} welcomed",  # 75%
        "when {name1} looked happy, {name2} welcomed",  # 75%
        "when {name1} felt hurt, {name2} welcomed",  # 75%
        "when {name1} was learning, {name2} welcomed",  # 74%
        "when {name1} came inside, {name2} welcomed",  # 74%
        "when {name1} visited, {name2} welcomed",  # 74%
        "when {name1} needed help, {name2} welcomed",  # 74%
        "when {name1} felt upset, {name2} welcomed",  # 74%
        "when {name1} seemed sad, {name2} welcomed",  # 74%
        "when {name1} was upset, {name2} welcomed",  # 73%
        "when {name1} stopped, {name2} welcomed",  # 73%
        "when {name1} seemed curious, {name2} welcomed",  # 73%
        "when {name1} shouted, {name2} welcomed",  # 73%
        "when {name1} came back, {name2} welcomed",  # 73%
        "when {name1} seemed confused, {name2} welcomed",  # 73%
        "when {name1} seemed tired, {name2} welcomed",  # 73%
        "when {name1} struggled, {name2} welcomed",  # 72%
        "when {name1} stopped by, {name2} welcomed",  # 72%
        "when {name1} showed up, {name2} welcomed",  # 71%
        "when {name1} forgot, {name2} welcomed",  # 71%
        "when {name1} seemed hurt, {name2} welcomed",  # 71%
        "when {name1} was happy, {name2} welcomed",  # 71%
        "when {name1} felt happy, {name2} welcomed",  # 70%
        "when {name1} returned, {name2} welcomed",  # 69%
        "when {name1} seemed upset, {name2} welcomed",  # 69%
        "when {name1} tripped, {name2} welcomed",  # 69%
        "when {name1} came home, {name2} welcomed",  # 68%
        "when {name1} wept, {name2} welcomed",  # 52%
        
        # ==================== BECKONED (108 templates, 54-75%) ====================
        "when {name1} looked afraid, {name2} beckoned",  # 75%
        "when {name1} felt lost, {name2} beckoned",  # 75%
        "when {name1} felt afraid, {name2} beckoned",  # 75%
        "when {name1} waited, {name2} beckoned",  # 74%
        "when {name1} was alone, {name2} beckoned",  # 74%
        "when {name1} was afraid, {name2} beckoned",  # 74%
        "when {name1} felt alone, {name2} beckoned",  # 74%
        "when {name1} stood alone, {name2} beckoned",  # 74%
        "when {name1} was lost, {name2} beckoned",  # 74%
        "when {name1} looked angry, {name2} beckoned",  # 73%
        "when {name1} felt angry, {name2} beckoned",  # 72%
        "when {name1} seemed afraid, {name2} beckoned",  # 72%
        "when {name1} sat alone, {name2} beckoned",  # 72%
        "when {name1} felt tired, {name2} beckoned",  # 72%
        "when {name1} wondered, {name2} beckoned",  # 72%
        "when {name1} was angry, {name2} beckoned",  # 72%
        "when {name1} felt confused, {name2} beckoned",  # 72%
        "when {name1} looked sick, {name2} beckoned",  # 72%
        "when {name1} looked lost, {name2} beckoned",  # 72%
        "when {name1} looked alone, {name2} beckoned",  # 72%
        "when {name1} seemed worried, {name2} beckoned",  # 72%
        "when {name1} felt hurt, {name2} beckoned",  # 71%
        "when {name1} felt worried, {name2} beckoned",  # 71%
        "when {name1} looked worried, {name2} beckoned",  # 71%
        "when {name1} was tired, {name2} beckoned",  # 71%
        "when {name1} felt nervous, {name2} beckoned",  # 71%
        "when {name1} was sick, {name2} beckoned",  # 71%
        "when {name1} was stuck, {name2} beckoned",  # 70%
        "when {name1} was learning, {name2} beckoned",  # 70%
        "when {name1} was hungry, {name2} beckoned",  # 70%
        "when {name1} fell, {name2} beckoned",  # 70%
        "when {name1} slipped, {name2} beckoned",  # 70%
        "when {name1} was scared, {name2} beckoned",  # 70%
        "when {name1} felt happy, {name2} beckoned",  # 70%
        "when {name1} felt cold, {name2} beckoned",  # 70%
        "when {name1} felt sad, {name2} beckoned",  # 70%
        "when {name1} was sad, {name2} beckoned",  # 70%
        "when {name1} was worried, {name2} beckoned",  # 70%
        "when {name1} felt scared, {name2} beckoned",  # 70%
        "when {name1} felt sick, {name2} beckoned",  # 70%
        "when {name1} walked in, {name2} beckoned",  # 69%
        "when {name1} looked hungry, {name2} beckoned",  # 69%
        "when {name1} looked scared, {name2} beckoned",  # 69%
        "when {name1} felt hungry, {name2} beckoned",  # 69%
        "when {name1} looked tired, {name2} beckoned",  # 69%
        "when {name1} stumbled, {name2} beckoned",  # 69%
        "when {name1} seemed angry, {name2} beckoned",  # 68%
        "when {name1} looked nervous, {name2} beckoned",  # 68%
        "when {name1} was nervous, {name2} beckoned",  # 68%
        "when {name1} seemed alone, {name2} beckoned",  # 68%
        "when {name1} got lost, {name2} beckoned",  # 68%
        "when {name1} sobbed, {name2} beckoned",  # 68%
        "when {name1} looked happy, {name2} beckoned",  # 67%
        "when {name1} was happy, {name2} beckoned",  # 67%
        "when {name1} called, {name2} beckoned",  # 67%
        "when {name1} seemed scared, {name2} beckoned",  # 67%
        "when {name1} looked hurt, {name2} beckoned",  # 67%
        "when {name1} walked inside, {name2} beckoned",  # 67%
        "when {name1} seemed lost, {name2} beckoned",  # 67%
        "when {name1} struggled, {name2} beckoned",  # 67%
        "when {name1} looked sad, {name2} beckoned",  # 67%
        "when {name1} called out, {name2} beckoned",  # 66%
        "when {name1} felt upset, {name2} beckoned",  # 66%
        "when {name1} was cold, {name2} beckoned",  # 66%
        "when {name1} felt curious, {name2} beckoned",  # 66%
        "when {name1} looked cold, {name2} beckoned",  # 66%
        "when {name1} was interested, {name2} beckoned",  # 65%
        "when {name1} seemed tired, {name2} beckoned",  # 65%
        "when {name1} forgot, {name2} beckoned",  # 65%
        "when {name1} came back, {name2} beckoned",  # 65%
        "when {name1} entered, {name2} beckoned",  # 65%
        "when {name1} looked bored, {name2} beckoned",  # 65%
        "when {name1} got scared, {name2} beckoned",  # 64%
        "when {name1} arrived, {name2} beckoned",  # 64%
        "when {name1} stood there, {name2} beckoned",  # 64%
        "when {name1} felt bored, {name2} beckoned",  # 64%
        "when {name1} seemed sad, {name2} beckoned",  # 64%
        "when {name1} came in, {name2} beckoned",  # 64%
        "when {name1} was hurt, {name2} beckoned",  # 64%
        "when {name1} cried, {name2} beckoned",  # 64%
        "when {name1} needed help, {name2} beckoned",  # 63%
        "when {name1} stepped in, {name2} beckoned",  # 63%
        "when {name1} showed up, {name2} beckoned",  # 63%
        "when {name1} hesitated, {name2} beckoned",  # 62%
        "when {name1} returned, {name2} beckoned",  # 62%
        "when {name1} appeared, {name2} beckoned",  # 62%
        "when {name1} was bored, {name2} beckoned",  # 62%
        "when {name1} looked confused, {name2} beckoned",  # 62%
        "when {name1} asked, {name2} beckoned",  # 62%
        "when {name1} was confused, {name2} beckoned",  # 62%
        "when {name1} stepped inside, {name2} beckoned",  # 62%
        "when {name1} stopped, {name2} beckoned",  # 61%
        "when {name1} looked upset, {name2} beckoned",  # 61%
        "when {name1} came home, {name2} beckoned",  # 61%
        "when {name1} came over, {name2} beckoned",  # 60%
        "when {name1} was upset, {name2} beckoned",  # 60%
        "when {name1} seemed hurt, {name2} beckoned",  # 60%
        "when {name1} came inside, {name2} beckoned",  # 60%
        "when {name1} stopped by, {name2} beckoned",  # 59%
        "when {name1} was curious, {name2} beckoned",  # 59%
        "when {name1} seemed bored, {name2} beckoned",  # 59%
        "when {name1} seemed confused, {name2} beckoned",  # 59%
        "when {name1} shouted, {name2} beckoned",  # 59%
        "when {name1} visited, {name2} beckoned",  # 58%
        "when {name1} tripped, {name2} beckoned",  # 57%
        "when {name1} seemed upset, {name2} beckoned",  # 56%
        "when {name1} paused, {name2} beckoned",  # 56%
        "when {name1} seemed curious, {name2} beckoned",  # 54%
        
        # ==================== REMINDED (108 templates, 55-78%) ====================
        "when {name1} walked inside, {name2} reminded",  # 78%
        "when {name1} entered, {name2} reminded",  # 78%
        "when {name1} felt bored, {name2} reminded",  # 76%
        "when {name1} looked lost, {name2} reminded",  # 76%
        "when {name1} looked worried, {name2} reminded",  # 75%
        "when {name1} arrived, {name2} reminded",  # 75%
        "when {name1} felt lost, {name2} reminded",  # 74%
        "when {name1} felt tired, {name2} reminded",  # 74%
        "when {name1} looked bored, {name2} reminded",  # 74%
        "when {name1} stood alone, {name2} reminded",  # 74%
        "when {name1} felt afraid, {name2} reminded",  # 74%
        "when {name1} looked afraid, {name2} reminded",  # 73%
        "when {name1} waited, {name2} reminded",  # 73%
        "when {name1} was bored, {name2} reminded",  # 73%
        "when {name1} walked in, {name2} reminded",  # 73%
        "when {name1} looked alone, {name2} reminded",  # 73%
        "when {name1} looked sick, {name2} reminded",  # 73%
        "when {name1} was lost, {name2} reminded",  # 72%
        "when {name1} slipped, {name2} reminded",  # 72%
        "when {name1} felt worried, {name2} reminded",  # 72%
        "when {name1} seemed worried, {name2} reminded",  # 72%
        "when {name1} looked tired, {name2} reminded",  # 72%
        "when {name1} stumbled, {name2} reminded",  # 72%
        "when {name1} felt confused, {name2} reminded",  # 72%
        "when {name1} looked hurt, {name2} reminded",  # 71%
        "when {name1} felt angry, {name2} reminded",  # 71%
        "when {name1} was afraid, {name2} reminded",  # 71%
        "when {name1} looked hungry, {name2} reminded",  # 71%
        "when {name1} stepped in, {name2} reminded",  # 71%
        "when {name1} felt alone, {name2} reminded",  # 71%
        "when {name1} stood there, {name2} reminded",  # 70%
        "when {name1} seemed lost, {name2} reminded",  # 70%
        "when {name1} called out, {name2} reminded",  # 70%
        "when {name1} was hungry, {name2} reminded",  # 70%
        "when {name1} was worried, {name2} reminded",  # 70%
        "when {name1} wondered, {name2} reminded",  # 70%
        "when {name1} stepped inside, {name2} reminded",  # 70%
        "when {name1} felt hungry, {name2} reminded",  # 70%
        "when {name1} was alone, {name2} reminded",  # 70%
        "when {name1} sobbed, {name2} reminded",  # 70%
        "when {name1} felt cold, {name2} reminded",  # 70%
        "when {name1} was interested, {name2} reminded",  # 70%
        "when {name1} was tired, {name2} reminded",  # 70%
        "when {name1} felt curious, {name2} reminded",  # 70%
        "when {name1} looked cold, {name2} reminded",  # 70%
        "when {name1} felt sick, {name2} reminded",  # 70%
        "when {name1} looked angry, {name2} reminded",  # 70%
        "when {name1} appeared, {name2} reminded",  # 70%
        "when {name1} came over, {name2} reminded",  # 69%
        "when {name1} looked confused, {name2} reminded",  # 69%
        "when {name1} felt nervous, {name2} reminded",  # 69%
        "when {name1} seemed bored, {name2} reminded",  # 68%
        "when {name1} sat alone, {name2} reminded",  # 68%
        "when {name1} was learning, {name2} reminded",  # 68%
        "when {name1} was sick, {name2} reminded",  # 68%
        "when {name1} seemed tired, {name2} reminded",  # 68%
        "when {name1} was cold, {name2} reminded",  # 68%
        "when {name1} visited, {name2} reminded",  # 67%
        "when {name1} was angry, {name2} reminded",  # 67%
        "when {name1} was scared, {name2} reminded",  # 67%
        "when {name1} struggled, {name2} reminded",  # 67%
        "when {name1} seemed angry, {name2} reminded",  # 67%
        "when {name1} was confused, {name2} reminded",  # 67%
        "when {name1} seemed afraid, {name2} reminded",  # 67%
        "when {name1} cried, {name2} reminded",  # 66%
        "when {name1} showed up, {name2} reminded",  # 66%
        "when {name1} called, {name2} reminded",  # 66%
        "when {name1} got lost, {name2} reminded",  # 66%
        "when {name1} felt hurt, {name2} reminded",  # 66%
        "when {name1} hesitated, {name2} reminded",  # 66%
        "when {name1} seemed alone, {name2} reminded",  # 66%
        "when {name1} returned, {name2} reminded",  # 66%
        "when {name1} felt happy, {name2} reminded",  # 66%
        "when {name1} came home, {name2} reminded",  # 66%
        "when {name1} fell, {name2} reminded",  # 66%
        "when {name1} shouted, {name2} reminded",  # 65%
        "when {name1} came in, {name2} reminded",  # 65%
        "when {name1} looked scared, {name2} reminded",  # 65%
        "when {name1} asked, {name2} reminded",  # 65%
        "when {name1} came inside, {name2} reminded",  # 65%
        "when {name1} paused, {name2} reminded",  # 65%
        "when {name1} looked upset, {name2} reminded",  # 65%
        "when {name1} stopped by, {name2} reminded",  # 64%
        "when {name1} looked happy, {name2} reminded",  # 64%
        "when {name1} seemed confused, {name2} reminded",  # 64%
        "when {name1} felt scared, {name2} reminded",  # 64%
        "when {name1} was stuck, {name2} reminded",  # 64%
        "when {name1} came back, {name2} reminded",  # 64%
        "when {name1} was hurt, {name2} reminded",  # 64%
        "when {name1} was curious, {name2} reminded",  # 63%
        "when {name1} stopped, {name2} reminded",  # 63%
        "when {name1} was happy, {name2} reminded",  # 63%
        "when {name1} felt upset, {name2} reminded",  # 63%
        "when {name1} looked nervous, {name2} reminded",  # 63%
        "when {name1} was sad, {name2} reminded",  # 62%
        "when {name1} forgot, {name2} reminded",  # 62%
        "when {name1} seemed hurt, {name2} reminded",  # 62%
        "when {name1} needed help, {name2} reminded",  # 62%
        "when {name1} seemed curious, {name2} reminded",  # 61%
        "when {name1} tripped, {name2} reminded",  # 60%
        "when {name1} was nervous, {name2} reminded",  # 60%
        "when {name1} seemed scared, {name2} reminded",  # 60%
        "when {name1} was upset, {name2} reminded",  # 60%
        "when {name1} felt sad, {name2} reminded",  # 60%
        "when {name1} got scared, {name2} reminded",  # 59%
        "when {name1} seemed upset, {name2} reminded",  # 58%
        "when {name1} looked sad, {name2} reminded",  # 56%
        "when {name1} seemed sad, {name2} reminded",  # 55%
        
        # ==================== TAUGHT (106 templates, 50-73%) ====================
        "when {name1} walked in, {name2} taught",  # 73%
        "when {name1} felt bored, {name2} taught",  # 72%
        "when {name1} walked inside, {name2} taught",  # 72%
        "when {name1} felt lost, {name2} taught",  # 71%
        "when {name1} looked lost, {name2} taught",  # 70%
        "when {name1} felt afraid, {name2} taught",  # 70%
        "when {name1} felt cold, {name2} taught",  # 69%
        "when {name1} felt alone, {name2} taught",  # 69%
        "when {name1} looked bored, {name2} taught",  # 69%
        "when {name1} stepped inside, {name2} taught",  # 68%
        "when {name1} felt curious, {name2} taught",  # 68%
        "when {name1} looked afraid, {name2} taught",  # 67%
        "when {name1} seemed bored, {name2} taught",  # 67%
        "when {name1} felt worried, {name2} taught",  # 67%
        "when {name1} seemed lost, {name2} taught",  # 67%
        "when {name1} was lost, {name2} taught",  # 67%
        "when {name1} seemed afraid, {name2} taught",  # 66%
        "when {name1} felt tired, {name2} taught",  # 66%
        "when {name1} stepped in, {name2} taught",  # 66%
        "when {name1} was bored, {name2} taught",  # 66%
        "when {name1} stood alone, {name2} taught",  # 66%
        "when {name1} looked alone, {name2} taught",  # 65%
        "when {name1} stood there, {name2} taught",  # 65%
        "when {name1} felt confused, {name2} taught",  # 64%
        "when {name1} felt scared, {name2} taught",  # 64%
        "when {name1} looked cold, {name2} taught",  # 64%
        "when {name1} looked worried, {name2} taught",  # 64%
        "when {name1} seemed alone, {name2} taught",  # 64%
        "when {name1} felt angry, {name2} taught",  # 64%
        "when {name1} felt upset, {name2} taught",  # 64%
        "when {name1} looked tired, {name2} taught",  # 63%
        "when {name1} waited, {name2} taught",  # 63%
        "when {name1} entered, {name2} taught",  # 63%
        "when {name1} felt sick, {name2} taught",  # 63%
        "when {name1} sat alone, {name2} taught",  # 63%
        "when {name1} was interested, {name2} taught",  # 63%
        "when {name1} felt nervous, {name2} taught",  # 63%
        "when {name1} was alone, {name2} taught",  # 63%
        "when {name1} seemed curious, {name2} taught",  # 63%
        "when {name1} felt sad, {name2} taught",  # 63%
        "when {name1} felt hurt, {name2} taught",  # 63%
        "when {name1} wondered, {name2} taught",  # 62%
        "when {name1} stumbled, {name2} taught",  # 62%
        "when {name1} looked sick, {name2} taught",  # 62%
        "when {name1} slipped, {name2} taught",  # 62%
        "when {name1} was afraid, {name2} taught",  # 62%
        "when {name1} called out, {name2} taught",  # 62%
        "when {name1} felt happy, {name2} taught",  # 62%
        "when {name1} looked sad, {name2} taught",  # 62%
        "when {name1} felt hungry, {name2} taught",  # 62%
        "when {name1} was curious, {name2} taught",  # 62%
        "when {name1} looked confused, {name2} taught",  # 61%
        "when {name1} seemed worried, {name2} taught",  # 61%
        "when {name1} sobbed, {name2} taught",  # 61%
        "when {name1} looked hurt, {name2} taught",  # 61%
        "when {name1} was worried, {name2} taught",  # 61%
        "when {name1} looked angry, {name2} taught",  # 60%
        "when {name1} seemed scared, {name2} taught",  # 60%
        "when {name1} was cold, {name2} taught",  # 60%
        "when {name1} was tired, {name2} taught",  # 60%
        "when {name1} looked hungry, {name2} taught",  # 60%
        "when {name1} seemed tired, {name2} taught",  # 60%
        "when {name1} looked scared, {name2} taught",  # 60%
        "when {name1} was scared, {name2} taught",  # 59%
        "when {name1} appeared, {name2} taught",  # 59%
        "when {name1} was confused, {name2} taught",  # 59%
        "when {name1} was sad, {name2} taught",  # 59%
        "when {name1} paused, {name2} taught",  # 59%
        "when {name1} looked upset, {name2} taught",  # 59%
        "when {name1} seemed confused, {name2} taught",  # 58%
        "when {name1} got lost, {name2} taught",  # 58%
        "when {name1} was learning, {name2} taught",  # 58%
        "when {name1} seemed angry, {name2} taught",  # 58%
        "when {name1} was sick, {name2} taught",  # 58%
        "when {name1} seemed sad, {name2} taught",  # 58%
        "when {name1} hesitated, {name2} taught",  # 58%
        "when {name1} was upset, {name2} taught",  # 58%
        "when {name1} arrived, {name2} taught",  # 57%
        "when {name1} looked nervous, {name2} taught",  # 57%
        "when {name1} was angry, {name2} taught",  # 57%
        "when {name1} was nervous, {name2} taught",  # 56%
        "when {name1} came in, {name2} taught",  # 56%
        "when {name1} called, {name2} taught",  # 56%
        "when {name1} seemed hurt, {name2} taught",  # 55%
        "when {name1} was hungry, {name2} taught",  # 55%
        "when {name1} cried, {name2} taught",  # 55%
        "when {name1} seemed upset, {name2} taught",  # 55%
        "when {name1} looked happy, {name2} taught",  # 55%
        "when {name1} returned, {name2} taught",  # 55%
        "when {name1} came inside, {name2} taught",  # 55%
        "when {name1} visited, {name2} taught",  # 55%
        "when {name1} came over, {name2} taught",  # 54%
        "when {name1} was hurt, {name2} taught",  # 54%
        "when {name1} was stuck, {name2} taught",  # 54%
        "when {name1} came back, {name2} taught",  # 53%
        "when {name1} got scared, {name2} taught",  # 53%
        "when {name1} struggled, {name2} taught",  # 53%
        "when {name1} fell, {name2} taught",  # 53%
        "when {name1} stopped, {name2} taught",  # 53%
        "when {name1} asked, {name2} taught",  # 53%
        "when {name1} tripped, {name2} taught",  # 52%
        "when {name1} stopped by, {name2} taught",  # 52%
        "when {name1} shouted, {name2} taught",  # 51%
        "when {name1} was happy, {name2} taught",  # 51%
        "when {name1} showed up, {name2} taught",  # 50%
        "when {name1} forgot, {name2} taught",  # 50%
        
        # ==================== HANDED (61 templates, 50-63%) ====================
        "when {name1} walked in, {name2} handed",  # 63%
        "when {name1} waited, {name2} handed",  # 58%
        "when {name1} walked inside, {name2} handed",  # 58%
        "when {name1} wondered, {name2} handed",  # 58%
        "when {name1} called, {name2} handed",  # 57%
        "when {name1} felt worried, {name2} handed",  # 56%
        "when {name1} felt afraid, {name2} handed",  # 56%
        "when {name1} felt bored, {name2} handed",  # 56%
        "when {name1} looked bored, {name2} handed",  # 55%
        "when {name1} felt alone, {name2} handed",  # 55%
        "when {name1} seemed bored, {name2} handed",  # 55%
        "when {name1} stood alone, {name2} handed",  # 55%
        "when {name1} seemed tired, {name2} handed",  # 55%
        "when {name1} cried, {name2} handed",  # 55%
        "when {name1} felt lost, {name2} handed",  # 54%
        "when {name1} looked tired, {name2} handed",  # 54%
        "when {name1} felt tired, {name2} handed",  # 54%
        "when {name1} was bored, {name2} handed",  # 54%
        "when {name1} felt confused, {name2} handed",  # 54%
        "when {name1} felt cold, {name2} handed",  # 54%
        "when {name1} seemed alone, {name2} handed",  # 53%
        "when {name1} looked worried, {name2} handed",  # 53%
        "when {name1} called out, {name2} handed",  # 53%
        "when {name1} stepped in, {name2} handed",  # 53%
        "when {name1} felt angry, {name2} handed",  # 53%
        "when {name1} looked confused, {name2} handed",  # 53%
        "when {name1} looked lost, {name2} handed",  # 53%
        "when {name1} was alone, {name2} handed",  # 53%
        "when {name1} looked angry, {name2} handed",  # 53%
        "when {name1} seemed lost, {name2} handed",  # 52%
        "when {name1} was lost, {name2} handed",  # 52%
        "when {name1} looked alone, {name2} handed",  # 52%
        "when {name1} felt hungry, {name2} handed",  # 52%
        "when {name1} seemed hurt, {name2} handed",  # 52%
        "when {name1} hesitated, {name2} handed",  # 52%
        "when {name1} came in, {name2} handed",  # 51%
        "when {name1} felt hurt, {name2} handed",  # 51%
        "when {name1} seemed worried, {name2} handed",  # 51%
        "when {name1} looked afraid, {name2} handed",  # 51%
        "when {name1} slipped, {name2} handed",  # 51%
        "when {name1} seemed angry, {name2} handed",  # 51%
        "when {name1} was afraid, {name2} handed",  # 51%
        "when {name1} sat alone, {name2} handed",  # 51%
        "when {name1} looked cold, {name2} handed",  # 51%
        "when {name1} looked sick, {name2} handed",  # 51%
        "when {name1} shouted, {name2} handed",  # 51%
        "when {name1} stepped inside, {name2} handed",  # 51%
        "when {name1} seemed sad, {name2} handed",  # 51%
        "when {name1} seemed afraid, {name2} handed",  # 51%
        "when {name1} stood there, {name2} handed",  # 51%
        "when {name1} seemed confused, {name2} handed",  # 51%
        "when {name1} sobbed, {name2} handed",  # 51%
        "when {name1} felt scared, {name2} handed",  # 51%
        "when {name1} was tired, {name2} handed",  # 51%
        "when {name1} was confused, {name2} handed",  # 51%
        "when {name1} felt happy, {name2} handed",  # 50%
        "when {name1} looked hurt, {name2} handed",  # 50%
        "when {name1} asked, {name2} handed",  # 50%
        "when {name1} was learning, {name2} handed",  # 50%
        "when {name1} stopped, {name2} handed",  # 50%
        "when {name1} felt sad, {name2} handed",  # 50%
        
        # ==================== WARNED (50 templates, 50-63%) ====================
        "when {name1} arrived, {name2} warned",  # 63%
        "when {name1} stood alone, {name2} warned",  # 57%
        "when {name1} looked afraid, {name2} warned",  # 57%
        "when {name1} walked in, {name2} warned",  # 57%
        "when {name1} was scared, {name2} warned",  # 57%
        "when {name1} waited, {name2} warned",  # 57%
        "when {name1} stepped in, {name2} warned",  # 56%
        "when {name1} looked worried, {name2} warned",  # 56%
        "when {name1} walked inside, {name2} warned",  # 56%
        "when {name1} sat alone, {name2} warned",  # 56%
        "when {name1} sobbed, {name2} warned",  # 56%
        "when {name1} looked bored, {name2} warned",  # 56%
        "when {name1} called out, {name2} warned",  # 56%
        "when {name1} felt bored, {name2} warned",  # 55%
        "when {name1} stumbled, {name2} warned",  # 54%
        "when {name1} came over, {name2} warned",  # 54%
        "when {name1} felt afraid, {name2} warned",  # 54%
        "when {name1} wondered, {name2} warned",  # 53%
        "when {name1} looked scared, {name2} warned",  # 53%
        "when {name1} was sick, {name2} warned",  # 53%
        "when {name1} was worried, {name2} warned",  # 53%
        "when {name1} looked angry, {name2} warned",  # 53%
        "when {name1} asked, {name2} warned",  # 52%
        "when {name1} looked cold, {name2} warned",  # 52%
        "when {name1} got lost, {name2} warned",  # 52%
        "when {name1} called, {name2} warned",  # 52%
        "when {name1} felt tired, {name2} warned",  # 52%
        "when {name1} looked alone, {name2} warned",  # 52%
        "when {name1} stopped by, {name2} warned",  # 52%
        "when {name1} felt scared, {name2} warned",  # 52%
        "when {name1} looked lost, {name2} warned",  # 52%
        "when {name1} felt lost, {name2} warned",  # 52%
        "when {name1} stood there, {name2} warned",  # 52%
        "when {name1} looked sad, {name2} warned",  # 52%
        "when {name1} was angry, {name2} warned",  # 52%
        "when {name1} struggled, {name2} warned",  # 51%
        "when {name1} was sad, {name2} warned",  # 51%
        "when {name1} was hungry, {name2} warned",  # 51%
        "when {name1} was tired, {name2} warned",  # 51%
        "when {name1} seemed afraid, {name2} warned",  # 51%
        "when {name1} was alone, {name2} warned",  # 51%
        "when {name1} visited, {name2} warned",  # 51%
        "when {name1} was lost, {name2} warned",  # 51%
        "when {name1} looked confused, {name2} warned",  # 51%
        "when {name1} looked sick, {name2} warned",  # 51%
        "when {name1} showed up, {name2} warned",  # 51%
        "when {name1} was afraid, {name2} warned",  # 50%
        "when {name1} stopped, {name2} warned",  # 50%
        "when {name1} was cold, {name2} warned",  # 50%
        "when {name1} appeared, {name2} warned",  # 50%
        
        # ==================== TOLD (35 templates, 50-55%) ====================
        "when {name1} looked afraid, {name2} told",  # 55%
        "when {name1} looked lost, {name2} told",  # 55%
        "when {name1} seemed afraid, {name2} told",  # 55%
        "when {name1} seemed lost, {name2} told",  # 54%
        "when {name1} waited, {name2} told",  # 54%
        "when {name1} felt bored, {name2} told",  # 54%
        "when {name1} seemed bored, {name2} told",  # 54%
        "when {name1} looked bored, {name2} told",  # 54%
        "when {name1} wondered, {name2} told",  # 53%
        "when {name1} walked in, {name2} told",  # 53%
        "when {name1} looked hurt, {name2} told",  # 53%
        "when {name1} was afraid, {name2} told",  # 53%
        "when {name1} felt lost, {name2} told",  # 53%
        "when {name1} stood alone, {name2} told",  # 53%
        "when {name1} seemed worried, {name2} told",  # 53%
        "when {name1} looked tired, {name2} told",  # 53%
        "when {name1} was lost, {name2} told",  # 52%
        "when {name1} looked angry, {name2} told",  # 52%
        "when {name1} stumbled, {name2} told",  # 52%
        "when {name1} looked worried, {name2} told",  # 52%
        "when {name1} seemed scared, {name2} told",  # 51%
        "when {name1} was bored, {name2} told",  # 51%
        "when {name1} looked confused, {name2} told",  # 51%
        "when {name1} was worried, {name2} told",  # 51%
        "when {name1} was tired, {name2} told",  # 51%
        "when {name1} felt afraid, {name2} told",  # 51%
        "when {name1} looked cold, {name2} told",  # 51%
        "when {name1} felt angry, {name2} told",  # 51%
        "when {name1} stood there, {name2} told",  # 51%
        "when {name1} felt hungry, {name2} told",  # 50%
        "when {name1} looked sick, {name2} told",  # 50%
        "when {name1} seemed alone, {name2} told",  # 50%
        "when {name1} seemed tired, {name2} told",  # 50%
        "when {name1} looked scared, {name2} told",  # 50%
        "when {name1} felt worried, {name2} told",  # 50%
        
        # ==================== GAVE (10 templates, 50-54%) ====================
        "when {name1} felt bored, {name2} gave",  # 54%
        "when {name1} stepped in, {name2} gave",  # 54%
        "when {name1} felt afraid, {name2} gave",  # 53%
        "when {name1} stepped inside, {name2} gave",  # 52%
        "when {name1} felt tired, {name2} gave",  # 51%
        "when {name1} felt lost, {name2} gave",  # 51%
        "when {name1} seemed bored, {name2} gave",  # 51%
        "when {name1} felt alone, {name2} gave",  # 51%
        "when {name1} looked bored, {name2} gave",  # 50%
        "when {name1} seemed lost, {name2} gave",  # 50%
    ]
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed)
        self.split = split
        # Pre-compute pronoun tokens
        him_ids = self.tokenizer.encode(" him", add_special_tokens=False)
        her_ids = self.tokenizer.encode(" her", add_special_tokens=False)
        self.him_token = him_ids[0] if him_ids else self.tokenizer.unk_token_id or 0
        self.her_token = her_ids[0] if her_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"ioi_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Randomly decide if name1 (the one pronoun refers to) is male or female
        name1_is_male = self.rng.random() > 0.5
        
        if name1_is_male:
            # name1 is male, name2 is female
            name1 = self.rng.choice(self.MALE_NAMES)
            name2 = self.rng.choice(self.FEMALE_NAMES)
            correct_token = self.him_token
            incorrect_token = self.her_token
        else:
            # name1 is female, name2 is male
            name1 = self.rng.choice(self.FEMALE_NAMES)
            name2 = self.rng.choice(self.MALE_NAMES)
            correct_token = self.her_token
            incorrect_token = self.him_token
        
        # Build the sentence
        template = self.rng.choice(self.TEMPLATES)
        context = template.format(name1=name1, name2=name2)
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class IOIRelaxedTask(BinaryTask):
    """
    Indirect Object Identification (IOI) task - RELAXED criteria.
    
    Same structure as IOIStrictTask but with relaxed filtering:
    - P(correct) > 0.3 (instead of > 0.5)
    - Binary P(correct | {him, her}) > 0.8 (instead of implicit from P > 0.5)
    
    This allows many more diverse verbs including 'followed', 'guided', 'showed',
    'supported', 'urged', 'hugged', 'helped', 'invited', 'trusted', 'led', etc.
    
    Total: 1394 templates across 20 verbs.
    """
    
    MALE_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter"]
    FEMALE_NAMES = ["Mia", "Kim", "Rita", "Lily", "Maria"]
    
    # Templates with P(correct) > 0.3 AND binary_P(correct) > 0.8
    # 20 verbs, 1394 templates total
    TEMPLATES = [
        # welcomed (96 templates)
        "when {name1} looked scared, {name2} welcomed",
        "when {name1} looked lost, {name2} welcomed",
        "when {name1} looked afraid, {name2} welcomed",
        "when {name1} looked bored, {name2} welcomed",
        "when {name1} felt lost, {name2} welcomed",
        "when {name1} felt bored, {name2} welcomed",
        "when {name1} walked inside, {name2} welcomed",
        "when {name1} walked in, {name2} welcomed",
        "when {name1} looked worried, {name2} welcomed",
        "when {name1} called out, {name2} welcomed",
        "when {name1} was scared, {name2} welcomed",
        "when {name1} stumbled, {name2} welcomed",
        "when {name1} was bored, {name2} welcomed",
        "when {name1} looked hungry, {name2} welcomed",
        "when {name1} was lost, {name2} welcomed",
        "when {name1} looked alone, {name2} welcomed",
        "when {name1} was afraid, {name2} welcomed",
        "when {name1} looked cold, {name2} welcomed",
        "when {name1} felt afraid, {name2} welcomed",
        "when {name1} looked angry, {name2} welcomed",
        "when {name1} wondered, {name2} welcomed",
        "when {name1} looked sick, {name2} welcomed",
        "when {name1} stood alone, {name2} welcomed",
        "when {name1} looked sad, {name2} welcomed",
        "when {name1} entered, {name2} welcomed",
        "when {name1} arrived, {name2} welcomed",
        "when {name1} was sad, {name2} welcomed",
        "when {name1} sat alone, {name2} welcomed",
        "when {name1} looked tired, {name2} welcomed",
        "when {name1} was worried, {name2} welcomed",
        "when {name1} called, {name2} welcomed",
        "when {name1} looked nervous, {name2} welcomed",
        "when {name1} was hungry, {name2} welcomed",
        "when {name1} stood there, {name2} welcomed",
        "when {name1} seemed bored, {name2} welcomed",
        "when {name1} appeared, {name2} welcomed",
        "when {name1} looked confused, {name2} welcomed",
        "when {name1} was angry, {name2} welcomed",
        "when {name1} was interested, {name2} welcomed",
        "when {name1} seemed lost, {name2} welcomed",
        "when {name1} felt alone, {name2} welcomed",
        "when {name1} felt cold, {name2} welcomed",
        "when {name1} felt scared, {name2} welcomed",
        "when {name1} waited, {name2} welcomed",
        "when {name1} was alone, {name2} welcomed",
        "when {name1} felt worried, {name2} welcomed",
        "when {name1} stepped inside, {name2} welcomed",
        "when {name1} was sick, {name2} welcomed",
        "when {name1} was tired, {name2} welcomed",
        "when {name1} looked hurt, {name2} welcomed",
        "when {name1} felt angry, {name2} welcomed",
        "when {name1} seemed scared, {name2} welcomed",
        "when {name1} sobbed, {name2} welcomed",
        "when {name1} felt nervous, {name2} welcomed",
        "when {name1} felt tired, {name2} welcomed",
        "when {name1} was confused, {name2} welcomed",
        "when {name1} stepped in, {name2} welcomed",
        "when {name1} seemed angry, {name2} welcomed",
        "when {name1} seemed alone, {name2} welcomed",
        "when {name1} felt hungry, {name2} welcomed",
        "when {name1} fell, {name2} welcomed",
        "when {name1} slipped, {name2} welcomed",
        "when {name1} seemed worried, {name2} welcomed",
        "when {name1} asked, {name2} welcomed",
        "when {name1} cried, {name2} welcomed",
        "when {name1} hesitated, {name2} welcomed",
        "when {name1} felt curious, {name2} welcomed",
        "when {name1} was curious, {name2} welcomed",
        "when {name1} seemed sad, {name2} welcomed",
        "when {name1} felt confused, {name2} welcomed",
        "when {name1} was cold, {name2} welcomed",
        "when {name1} was hurt, {name2} welcomed",
        "when {name1} was nervous, {name2} welcomed",
        "when {name1} was stuck, {name2} welcomed",
        "when {name1} looked upset, {name2} welcomed",
        "when {name1} felt hurt, {name2} welcomed",
        "when {name1} felt sick, {name2} welcomed",
        "when {name1} felt sad, {name2} welcomed",
        "when {name1} was learning, {name2} welcomed",
        "when {name1} came inside, {name2} welcomed",
        "when {name1} was upset, {name2} welcomed",
        "when {name1} paused, {name2} welcomed",
        "when {name1} looked happy, {name2} welcomed",
        "when {name1} shouted, {name2} welcomed",
        "when {name1} felt upset, {name2} welcomed",
        "when {name1} stopped, {name2} welcomed",
        "when {name1} needed help, {name2} welcomed",
        "when {name1} came back, {name2} welcomed",
        "when {name1} seemed confused, {name2} welcomed",
        "when {name1} showed up, {name2} welcomed",
        "when {name1} was happy, {name2} welcomed",
        "when {name1} returned, {name2} welcomed",
        "when {name1} forgot, {name2} welcomed",
        "when {name1} came home, {name2} welcomed",
        "when {name1} tripped, {name2} welcomed",
        "when {name1} felt happy, {name2} welcomed",
        # reminded (96 templates)
        "when {name1} entered, {name2} reminded",
        "when {name1} walked inside, {name2} reminded",
        "when {name1} felt bored, {name2} reminded",
        "when {name1} arrived, {name2} reminded",
        "when {name1} felt lost, {name2} reminded",
        "when {name1} looked sick, {name2} reminded",
        "when {name1} waited, {name2} reminded",
        "when {name1} looked worried, {name2} reminded",
        "when {name1} looked lost, {name2} reminded",
        "when {name1} felt worried, {name2} reminded",
        "when {name1} looked hungry, {name2} reminded",
        "when {name1} looked afraid, {name2} reminded",
        "when {name1} walked in, {name2} reminded",
        "when {name1} seemed lost, {name2} reminded",
        "when {name1} stood alone, {name2} reminded",
        "when {name1} felt tired, {name2} reminded",
        "when {name1} was lost, {name2} reminded",
        "when {name1} felt afraid, {name2} reminded",
        "when {name1} looked cold, {name2} reminded",
        "when {name1} was worried, {name2} reminded",
        "when {name1} stepped in, {name2} reminded",
        "when {name1} looked tired, {name2} reminded",
        "when {name1} was bored, {name2} reminded",
        "when {name1} felt alone, {name2} reminded",
        "when {name1} stepped inside, {name2} reminded",
        "when {name1} looked bored, {name2} reminded",
        "when {name1} felt hungry, {name2} reminded",
        "when {name1} stumbled, {name2} reminded",
        "when {name1} looked angry, {name2} reminded",
        "when {name1} felt cold, {name2} reminded",
        "when {name1} was afraid, {name2} reminded",
        "when {name1} wondered, {name2} reminded",
        "when {name1} looked confused, {name2} reminded",
        "when {name1} seemed worried, {name2} reminded",
        "when {name1} felt confused, {name2} reminded",
        "when {name1} felt angry, {name2} reminded",
        "when {name1} stood there, {name2} reminded",
        "when {name1} looked hurt, {name2} reminded",
        "when {name1} sobbed, {name2} reminded",
        "when {name1} looked alone, {name2} reminded",
        "when {name1} appeared, {name2} reminded",
        "when {name1} felt curious, {name2} reminded",
        "when {name1} was alone, {name2} reminded",
        "when {name1} slipped, {name2} reminded",
        "when {name1} was sick, {name2} reminded",
        "when {name1} was confused, {name2} reminded",
        "when {name1} sat alone, {name2} reminded",
        "when {name1} was angry, {name2} reminded",
        "when {name1} was tired, {name2} reminded",
        "when {name1} was interested, {name2} reminded",
        "when {name1} asked, {name2} reminded",
        "when {name1} felt hurt, {name2} reminded",
        "when {name1} seemed bored, {name2} reminded",
        "when {name1} looked scared, {name2} reminded",
        "when {name1} called out, {name2} reminded",
        "when {name1} stopped, {name2} reminded",
        "when {name1} was cold, {name2} reminded",
        "when {name1} seemed alone, {name2} reminded",
        "when {name1} was learning, {name2} reminded",
        "when {name1} came back, {name2} reminded",
        "when {name1} was hungry, {name2} reminded",
        "when {name1} was scared, {name2} reminded",
        "when {name1} fell, {name2} reminded",
        "when {name1} paused, {name2} reminded",
        "when {name1} felt sick, {name2} reminded",
        "when {name1} felt upset, {name2} reminded",
        "when {name1} cried, {name2} reminded",
        "when {name1} called, {name2} reminded",
        "when {name1} showed up, {name2} reminded",
        "when {name1} felt scared, {name2} reminded",
        "when {name1} was hurt, {name2} reminded",
        "when {name1} felt nervous, {name2} reminded",
        "when {name1} looked happy, {name2} reminded",
        "when {name1} felt happy, {name2} reminded",
        "when {name1} looked upset, {name2} reminded",
        "when {name1} shouted, {name2} reminded",
        "when {name1} was curious, {name2} reminded",
        "when {name1} came inside, {name2} reminded",
        "when {name1} seemed angry, {name2} reminded",
        "when {name1} hesitated, {name2} reminded",
        "when {name1} returned, {name2} reminded",
        "when {name1} was stuck, {name2} reminded",
        "when {name1} forgot, {name2} reminded",
        "when {name1} seemed scared, {name2} reminded",
        "when {name1} was upset, {name2} reminded",
        "when {name1} needed help, {name2} reminded",
        "when {name1} was sad, {name2} reminded",
        "when {name1} came home, {name2} reminded",
        "when {name1} tripped, {name2} reminded",
        "when {name1} looked nervous, {name2} reminded",
        "when {name1} seemed confused, {name2} reminded",
        "when {name1} was happy, {name2} reminded",
        "when {name1} felt sad, {name2} reminded",
        "when {name1} was nervous, {name2} reminded",
        "when {name1} looked sad, {name2} reminded",
        "when {name1} seemed sad, {name2} reminded",
        # taught (96 templates)
        "when {name1} walked in, {name2} taught",
        "when {name1} felt bored, {name2} taught",
        "when {name1} walked inside, {name2} taught",
        "when {name1} felt lost, {name2} taught",
        "when {name1} looked lost, {name2} taught",
        "when {name1} felt alone, {name2} taught",
        "when {name1} felt afraid, {name2} taught",
        "when {name1} felt cold, {name2} taught",
        "when {name1} stepped inside, {name2} taught",
        "when {name1} looked bored, {name2} taught",
        "when {name1} felt curious, {name2} taught",
        "when {name1} was lost, {name2} taught",
        "when {name1} seemed bored, {name2} taught",
        "when {name1} felt tired, {name2} taught",
        "when {name1} felt worried, {name2} taught",
        "when {name1} seemed lost, {name2} taught",
        "when {name1} stood alone, {name2} taught",
        "when {name1} looked afraid, {name2} taught",
        "when {name1} stepped in, {name2} taught",
        "when {name1} was bored, {name2} taught",
        "when {name1} stood there, {name2} taught",
        "when {name1} felt confused, {name2} taught",
        "when {name1} wondered, {name2} taught",
        "when {name1} looked worried, {name2} taught",
        "when {name1} looked alone, {name2} taught",
        "when {name1} looked cold, {name2} taught",
        "when {name1} felt angry, {name2} taught",
        "when {name1} looked tired, {name2} taught",
        "when {name1} seemed alone, {name2} taught",
        "when {name1} felt sick, {name2} taught",
        "when {name1} felt hurt, {name2} taught",
        "when {name1} felt scared, {name2} taught",
        "when {name1} sat alone, {name2} taught",
        "when {name1} felt nervous, {name2} taught",
        "when {name1} seemed worried, {name2} taught",
        "when {name1} waited, {name2} taught",
        "when {name1} was afraid, {name2} taught",
        "when {name1} felt upset, {name2} taught",
        "when {name1} looked hurt, {name2} taught",
        "when {name1} felt sad, {name2} taught",
        "when {name1} was interested, {name2} taught",
        "when {name1} entered, {name2} taught",
        "when {name1} looked sick, {name2} taught",
        "when {name1} called out, {name2} taught",
        "when {name1} stumbled, {name2} taught",
        "when {name1} was curious, {name2} taught",
        "when {name1} was alone, {name2} taught",
        "when {name1} felt happy, {name2} taught",
        "when {name1} looked sad, {name2} taught",
        "when {name1} felt hungry, {name2} taught",
        "when {name1} slipped, {name2} taught",
        "when {name1} looked angry, {name2} taught",
        "when {name1} sobbed, {name2} taught",
        "when {name1} looked confused, {name2} taught",
        "when {name1} was tired, {name2} taught",
        "when {name1} was worried, {name2} taught",
        "when {name1} looked scared, {name2} taught",
        "when {name1} was scared, {name2} taught",
        "when {name1} seemed scared, {name2} taught",
        "when {name1} seemed sad, {name2} taught",
        "when {name1} looked hungry, {name2} taught",
        "when {name1} was confused, {name2} taught",
        "when {name1} was cold, {name2} taught",
        "when {name1} appeared, {name2} taught",
        "when {name1} paused, {name2} taught",
        "when {name1} seemed confused, {name2} taught",
        "when {name1} looked upset, {name2} taught",
        "when {name1} hesitated, {name2} taught",
        "when {name1} was sad, {name2} taught",
        "when {name1} looked nervous, {name2} taught",
        "when {name1} was learning, {name2} taught",
        "when {name1} seemed angry, {name2} taught",
        "when {name1} was angry, {name2} taught",
        "when {name1} was sick, {name2} taught",
        "when {name1} was upset, {name2} taught",
        "when {name1} arrived, {name2} taught",
        "when {name1} was nervous, {name2} taught",
        "when {name1} came inside, {name2} taught",
        "when {name1} called, {name2} taught",
        "when {name1} looked happy, {name2} taught",
        "when {name1} returned, {name2} taught",
        "when {name1} was hungry, {name2} taught",
        "when {name1} cried, {name2} taught",
        "when {name1} was hurt, {name2} taught",
        "when {name1} came back, {name2} taught",
        "when {name1} stopped, {name2} taught",
        "when {name1} fell, {name2} taught",
        "when {name1} was stuck, {name2} taught",
        "when {name1} asked, {name2} taught",
        "when {name1} tripped, {name2} taught",
        "when {name1} forgot, {name2} taught",
        "when {name1} was happy, {name2} taught",
        "when {name1} shouted, {name2} taught",
        "when {name1} showed up, {name2} taught",
        "when {name1} came home, {name2} taught",
        "when {name1} needed help, {name2} taught",
        # handed (96 templates)
        "when {name1} walked in, {name2} handed",
        "when {name1} wondered, {name2} handed",
        "when {name1} called, {name2} handed",
        "when {name1} felt worried, {name2} handed",
        "when {name1} felt afraid, {name2} handed",
        "when {name1} walked inside, {name2} handed",
        "when {name1} cried, {name2} handed",
        "when {name1} felt bored, {name2} handed",
        "when {name1} waited, {name2} handed",
        "when {name1} seemed bored, {name2} handed",
        "when {name1} felt lost, {name2} handed",
        "when {name1} seemed worried, {name2} handed",
        "when {name1} looked bored, {name2} handed",
        "when {name1} looked tired, {name2} handed",
        "when {name1} looked lost, {name2} handed",
        "when {name1} called out, {name2} handed",
        "when {name1} seemed lost, {name2} handed",
        "when {name1} felt angry, {name2} handed",
        "when {name1} was bored, {name2} handed",
        "when {name1} looked confused, {name2} handed",
        "when {name1} felt tired, {name2} handed",
        "when {name1} stood alone, {name2} handed",
        "when {name1} was alone, {name2} handed",
        "when {name1} looked alone, {name2} handed",
        "when {name1} shouted, {name2} handed",
        "when {name1} felt cold, {name2} handed",
        "when {name1} stepped in, {name2} handed",
        "when {name1} felt alone, {name2} handed",
        "when {name1} looked cold, {name2} handed",
        "when {name1} seemed confused, {name2} handed",
        "when {name1} seemed alone, {name2} handed",
        "when {name1} stood there, {name2} handed",
        "when {name1} looked angry, {name2} handed",
        "when {name1} looked hurt, {name2} handed",
        "when {name1} was tired, {name2} handed",
        "when {name1} asked, {name2} handed",
        "when {name1} sobbed, {name2} handed",
        "when {name1} sat alone, {name2} handed",
        "when {name1} felt confused, {name2} handed",
        "when {name1} felt sick, {name2} handed",
        "when {name1} was lost, {name2} handed",
        "when {name1} was worried, {name2} handed",
        "when {name1} looked sick, {name2} handed",
        "when {name1} seemed angry, {name2} handed",
        "when {name1} seemed sad, {name2} handed",
        "when {name1} felt hurt, {name2} handed",
        "when {name1} looked worried, {name2} handed",
        "when {name1} looked afraid, {name2} handed",
        "when {name1} stepped inside, {name2} handed",
        "when {name1} hesitated, {name2} handed",
        "when {name1} felt scared, {name2} handed",
        "when {name1} felt nervous, {name2} handed",
        "when {name1} slipped, {name2} handed",
        "when {name1} stopped, {name2} handed",
        "when {name1} looked scared, {name2} handed",
        "when {name1} was interested, {name2} handed",
        "when {name1} looked happy, {name2} handed",
        "when {name1} was afraid, {name2} handed",
        "when {name1} was scared, {name2} handed",
        "when {name1} fell, {name2} handed",
        "when {name1} tripped, {name2} handed",
        "when {name1} was learning, {name2} handed",
        "when {name1} was confused, {name2} handed",
        "when {name1} felt sad, {name2} handed",
        "when {name1} paused, {name2} handed",
        "when {name1} felt happy, {name2} handed",
        "when {name1} appeared, {name2} handed",
        "when {name1} returned, {name2} handed",
        "when {name1} was sad, {name2} handed",
        "when {name1} came back, {name2} handed",
        "when {name1} seemed scared, {name2} handed",
        "when {name1} was cold, {name2} handed",
        "when {name1} looked sad, {name2} handed",
        "when {name1} looked hungry, {name2} handed",
        "when {name1} looked upset, {name2} handed",
        "when {name1} stumbled, {name2} handed",
        "when {name1} was happy, {name2} handed",
        "when {name1} felt curious, {name2} handed",
        "when {name1} felt upset, {name2} handed",
        "when {name1} was hungry, {name2} handed",
        "when {name1} felt hungry, {name2} handed",
        "when {name1} looked nervous, {name2} handed",
        "when {name1} forgot, {name2} handed",
        "when {name1} was angry, {name2} handed",
        "when {name1} came inside, {name2} handed",
        "when {name1} was hurt, {name2} handed",
        "when {name1} was nervous, {name2} handed",
        "when {name1} entered, {name2} handed",
        "when {name1} was curious, {name2} handed",
        "when {name1} was sick, {name2} handed",
        "when {name1} was upset, {name2} handed",
        "when {name1} was stuck, {name2} handed",
        "when {name1} came home, {name2} handed",
        "when {name1} arrived, {name2} handed",
        "when {name1} showed up, {name2} handed",
        "when {name1} needed help, {name2} handed",
        # gave (96 templates)
        "when {name1} stepped in, {name2} gave",
        "when {name1} stepped inside, {name2} gave",
        "when {name1} felt bored, {name2} gave",
        "when {name1} seemed bored, {name2} gave",
        "when {name1} felt lost, {name2} gave",
        "when {name1} looked lost, {name2} gave",
        "when {name1} felt alone, {name2} gave",
        "when {name1} walked in, {name2} gave",
        "when {name1} looked bored, {name2} gave",
        "when {name1} felt cold, {name2} gave",
        "when {name1} felt tired, {name2} gave",
        "when {name1} seemed lost, {name2} gave",
        "when {name1} felt worried, {name2} gave",
        "when {name1} was bored, {name2} gave",
        "when {name1} felt afraid, {name2} gave",
        "when {name1} felt confused, {name2} gave",
        "when {name1} felt hurt, {name2} gave",
        "when {name1} was lost, {name2} gave",
        "when {name1} felt curious, {name2} gave",
        "when {name1} seemed scared, {name2} gave",
        "when {name1} looked afraid, {name2} gave",
        "when {name1} felt scared, {name2} gave",
        "when {name1} looked alone, {name2} gave",
        "when {name1} stood alone, {name2} gave",
        "when {name1} felt nervous, {name2} gave",
        "when {name1} was afraid, {name2} gave",
        "when {name1} felt angry, {name2} gave",
        "when {name1} looked worried, {name2} gave",
        "when {name1} seemed alone, {name2} gave",
        "when {name1} looked scared, {name2} gave",
        "when {name1} waited, {name2} gave",
        "when {name1} felt sad, {name2} gave",
        "when {name1} called out, {name2} gave",
        "when {name1} was tired, {name2} gave",
        "when {name1} looked tired, {name2} gave",
        "when {name1} felt sick, {name2} gave",
        "when {name1} was sad, {name2} gave",
        "when {name1} walked inside, {name2} gave",
        "when {name1} was alone, {name2} gave",
        "when {name1} felt upset, {name2} gave",
        "when {name1} sobbed, {name2} gave",
        "when {name1} looked sad, {name2} gave",
        "when {name1} seemed worried, {name2} gave",
        "when {name1} cried, {name2} gave",
        "when {name1} was worried, {name2} gave",
        "when {name1} felt happy, {name2} gave",
        "when {name1} looked happy, {name2} gave",
        "when {name1} was interested, {name2} gave",
        "when {name1} seemed sad, {name2} gave",
        "when {name1} looked cold, {name2} gave",
        "when {name1} was learning, {name2} gave",
        "when {name1} shouted, {name2} gave",
        "when {name1} looked confused, {name2} gave",
        "when {name1} wondered, {name2} gave",
        "when {name1} was cold, {name2} gave",
        "when {name1} called, {name2} gave",
        "when {name1} looked upset, {name2} gave",
        "when {name1} appeared, {name2} gave",
        "when {name1} seemed confused, {name2} gave",
        "when {name1} looked hurt, {name2} gave",
        "when {name1} felt hungry, {name2} gave",
        "when {name1} was confused, {name2} gave",
        "when {name1} looked hungry, {name2} gave",
        "when {name1} paused, {name2} gave",
        "when {name1} looked nervous, {name2} gave",
        "when {name1} hesitated, {name2} gave",
        "when {name1} sat alone, {name2} gave",
        "when {name1} asked, {name2} gave",
        "when {name1} seemed angry, {name2} gave",
        "when {name1} was scared, {name2} gave",
        "when {name1} was hungry, {name2} gave",
        "when {name1} was angry, {name2} gave",
        "when {name1} was upset, {name2} gave",
        "when {name1} was nervous, {name2} gave",
        "when {name1} stumbled, {name2} gave",
        "when {name1} looked sick, {name2} gave",
        "when {name1} was hurt, {name2} gave",
        "when {name1} was happy, {name2} gave",
        "when {name1} looked angry, {name2} gave",
        "when {name1} was curious, {name2} gave",
        "when {name1} came back, {name2} gave",
        "when {name1} entered, {name2} gave",
        "when {name1} stood there, {name2} gave",
        "when {name1} returned, {name2} gave",
        "when {name1} arrived, {name2} gave",
        "when {name1} slipped, {name2} gave",
        "when {name1} was sick, {name2} gave",
        "when {name1} came home, {name2} gave",
        "when {name1} came inside, {name2} gave",
        "when {name1} fell, {name2} gave",
        "when {name1} stopped, {name2} gave",
        "when {name1} was stuck, {name2} gave",
        "when {name1} showed up, {name2} gave",
        "when {name1} forgot, {name2} gave",
        "when {name1} tripped, {name2} gave",
        "when {name1} needed help, {name2} gave",
        # warned (94 templates)
        "when {name1} arrived, {name2} warned",
        "when {name1} stood alone, {name2} warned",
        "when {name1} stepped in, {name2} warned",
        "when {name1} looked bored, {name2} warned",
        "when {name1} stumbled, {name2} warned",
        "when {name1} stopped, {name2} warned",
        "when {name1} walked inside, {name2} warned",
        "when {name1} looked afraid, {name2} warned",
        "when {name1} walked in, {name2} warned",
        "when {name1} called, {name2} warned",
        "when {name1} sobbed, {name2} warned",
        "when {name1} stood there, {name2} warned",
        "when {name1} waited, {name2} warned",
        "when {name1} was afraid, {name2} warned",
        "when {name1} asked, {name2} warned",
        "when {name1} felt afraid, {name2} warned",
        "when {name1} was worried, {name2} warned",
        "when {name1} looked worried, {name2} warned",
        "when {name1} sat alone, {name2} warned",
        "when {name1} was scared, {name2} warned",
        "when {name1} felt bored, {name2} warned",
        "when {name1} looked lost, {name2} warned",
        "when {name1} called out, {name2} warned",
        "when {name1} was lost, {name2} warned",
        "when {name1} looked alone, {name2} warned",
        "when {name1} was sick, {name2} warned",
        "when {name1} looked sick, {name2} warned",
        "when {name1} seemed scared, {name2} warned",
        "when {name1} was sad, {name2} warned",
        "when {name1} appeared, {name2} warned",
        "when {name1} was tired, {name2} warned",
        "when {name1} felt tired, {name2} warned",
        "when {name1} looked hungry, {name2} warned",
        "when {name1} looked cold, {name2} warned",
        "when {name1} felt scared, {name2} warned",
        "when {name1} wondered, {name2} warned",
        "when {name1} showed up, {name2} warned",
        "when {name1} was angry, {name2} warned",
        "when {name1} felt lost, {name2} warned",
        "when {name1} looked scared, {name2} warned",
        "when {name1} felt alone, {name2} warned",
        "when {name1} felt angry, {name2} warned",
        "when {name1} stepped inside, {name2} warned",
        "when {name1} felt sick, {name2} warned",
        "when {name1} looked sad, {name2} warned",
        "when {name1} entered, {name2} warned",
        "when {name1} was cold, {name2} warned",
        "when {name1} was learning, {name2} warned",
        "when {name1} looked upset, {name2} warned",
        "when {name1} was alone, {name2} warned",
        "when {name1} looked angry, {name2} warned",
        "when {name1} was hungry, {name2} warned",
        "when {name1} seemed worried, {name2} warned",
        "when {name1} needed help, {name2} warned",
        "when {name1} slipped, {name2} warned",
        "when {name1} was confused, {name2} warned",
        "when {name1} felt sad, {name2} warned",
        "when {name1} felt worried, {name2} warned",
        "when {name1} tripped, {name2} warned",
        "when {name1} looked hurt, {name2} warned",
        "when {name1} felt hungry, {name2} warned",
        "when {name1} was bored, {name2} warned",
        "when {name1} looked happy, {name2} warned",
        "when {name1} paused, {name2} warned",
        "when {name1} was happy, {name2} warned",
        "when {name1} was hurt, {name2} warned",
        "when {name1} looked tired, {name2} warned",
        "when {name1} returned, {name2} warned",
        "when {name1} felt nervous, {name2} warned",
        "when {name1} came home, {name2} warned",
        "when {name1} seemed alone, {name2} warned",
        "when {name1} seemed bored, {name2} warned",
        "when {name1} hesitated, {name2} warned",
        "when {name1} felt upset, {name2} warned",
        "when {name1} was upset, {name2} warned",
        "when {name1} felt happy, {name2} warned",
        "when {name1} seemed lost, {name2} warned",
        "when {name1} was interested, {name2} warned",
        "when {name1} felt confused, {name2} warned",
        "when {name1} was nervous, {name2} warned",
        "when {name1} was curious, {name2} warned",
        "when {name1} felt hurt, {name2} warned",
        "when {name1} came inside, {name2} warned",
        "when {name1} seemed angry, {name2} warned",
        "when {name1} fell, {name2} warned",
        "when {name1} looked confused, {name2} warned",
        "when {name1} felt cold, {name2} warned",
        "when {name1} felt curious, {name2} warned",
        "when {name1} cried, {name2} warned",
        "when {name1} looked nervous, {name2} warned",
        "when {name1} shouted, {name2} warned",
        "when {name1} forgot, {name2} warned",
        "when {name1} came back, {name2} warned",
        "when {name1} was stuck, {name2} warned",
        # guided (92 templates)
        "when {name1} walked in, {name2} guided",
        "when {name1} felt scared, {name2} guided",
        "when {name1} seemed scared, {name2} guided",
        "when {name1} felt afraid, {name2} guided",
        "when {name1} felt nervous, {name2} guided",
        "when {name1} was scared, {name2} guided",
        "when {name1} looked nervous, {name2} guided",
        "when {name1} looked afraid, {name2} guided",
        "when {name1} looked scared, {name2} guided",
        "when {name1} felt cold, {name2} guided",
        "when {name1} felt sick, {name2} guided",
        "when {name1} waited, {name2} guided",
        "when {name1} seemed bored, {name2} guided",
        "when {name1} seemed lost, {name2} guided",
        "when {name1} looked lost, {name2} guided",
        "when {name1} stepped in, {name2} guided",
        "when {name1} felt lost, {name2} guided",
        "when {name1} felt sad, {name2} guided",
        "when {name1} was afraid, {name2} guided",
        "when {name1} looked bored, {name2} guided",
        "when {name1} felt alone, {name2} guided",
        "when {name1} seemed alone, {name2} guided",
        "when {name1} stood alone, {name2} guided",
        "when {name1} felt upset, {name2} guided",
        "when {name1} looked cold, {name2} guided",
        "when {name1} was interested, {name2} guided",
        "when {name1} felt worried, {name2} guided",
        "when {name1} felt tired, {name2} guided",
        "when {name1} felt hurt, {name2} guided",
        "when {name1} looked worried, {name2} guided",
        "when {name1} felt bored, {name2} guided",
        "when {name1} was sad, {name2} guided",
        "when {name1} called out, {name2} guided",
        "when {name1} seemed angry, {name2} guided",
        "when {name1} called, {name2} guided",
        "when {name1} was nervous, {name2} guided",
        "when {name1} looked tired, {name2} guided",
        "when {name1} was tired, {name2} guided",
        "when {name1} looked hungry, {name2} guided",
        "when {name1} looked alone, {name2} guided",
        "when {name1} seemed sad, {name2} guided",
        "when {name1} looked confused, {name2} guided",
        "when {name1} felt confused, {name2} guided",
        "when {name1} sat alone, {name2} guided",
        "when {name1} slipped, {name2} guided",
        "when {name1} was bored, {name2} guided",
        "when {name1} felt angry, {name2} guided",
        "when {name1} looked sick, {name2} guided",
        "when {name1} was hungry, {name2} guided",
        "when {name1} felt curious, {name2} guided",
        "when {name1} was cold, {name2} guided",
        "when {name1} wondered, {name2} guided",
        "when {name1} was alone, {name2} guided",
        "when {name1} appeared, {name2} guided",
        "when {name1} entered, {name2} guided",
        "when {name1} felt hungry, {name2} guided",
        "when {name1} sobbed, {name2} guided",
        "when {name1} looked angry, {name2} guided",
        "when {name1} seemed worried, {name2} guided",
        "when {name1} was lost, {name2} guided",
        "when {name1} was sick, {name2} guided",
        "when {name1} hesitated, {name2} guided",
        "when {name1} looked happy, {name2} guided",
        "when {name1} arrived, {name2} guided",
        "when {name1} cried, {name2} guided",
        "when {name1} felt happy, {name2} guided",
        "when {name1} stepped inside, {name2} guided",
        "when {name1} walked inside, {name2} guided",
        "when {name1} was worried, {name2} guided",
        "when {name1} stopped, {name2} guided",
        "when {name1} was learning, {name2} guided",
        "when {name1} paused, {name2} guided",
        "when {name1} was angry, {name2} guided",
        "when {name1} seemed confused, {name2} guided",
        "when {name1} stood there, {name2} guided",
        "when {name1} looked sad, {name2} guided",
        "when {name1} looked hurt, {name2} guided",
        "when {name1} shouted, {name2} guided",
        "when {name1} stumbled, {name2} guided",
        "when {name1} looked upset, {name2} guided",
        "when {name1} was happy, {name2} guided",
        "when {name1} asked, {name2} guided",
        "when {name1} forgot, {name2} guided",
        "when {name1} fell, {name2} guided",
        "when {name1} was confused, {name2} guided",
        "when {name1} was curious, {name2} guided",
        "when {name1} returned, {name2} guided",
        "when {name1} was upset, {name2} guided",
        "when {name1} needed help, {name2} guided",
        "when {name1} came back, {name2} guided",
        "when {name1} was hurt, {name2} guided",
        "when {name1} came inside, {name2} guided",
        # showed (89 templates)
        "when {name1} walked in, {name2} showed",
        "when {name1} stood alone, {name2} showed",
        "when {name1} seemed scared, {name2} showed",
        "when {name1} felt lost, {name2} showed",
        "when {name1} waited, {name2} showed",
        "when {name1} felt worried, {name2} showed",
        "when {name1} looked bored, {name2} showed",
        "when {name1} looked scared, {name2} showed",
        "when {name1} felt alone, {name2} showed",
        "when {name1} asked, {name2} showed",
        "when {name1} was scared, {name2} showed",
        "when {name1} felt bored, {name2} showed",
        "when {name1} looked lost, {name2} showed",
        "when {name1} stepped in, {name2} showed",
        "when {name1} felt afraid, {name2} showed",
        "when {name1} felt tired, {name2} showed",
        "when {name1} seemed bored, {name2} showed",
        "when {name1} seemed lost, {name2} showed",
        "when {name1} walked inside, {name2} showed",
        "when {name1} felt scared, {name2} showed",
        "when {name1} looked alone, {name2} showed",
        "when {name1} seemed alone, {name2} showed",
        "when {name1} looked worried, {name2} showed",
        "when {name1} was lost, {name2} showed",
        "when {name1} was bored, {name2} showed",
        "when {name1} sat alone, {name2} showed",
        "when {name1} felt cold, {name2} showed",
        "when {name1} wondered, {name2} showed",
        "when {name1} felt sad, {name2} showed",
        "when {name1} looked sad, {name2} showed",
        "when {name1} felt curious, {name2} showed",
        "when {name1} called out, {name2} showed",
        "when {name1} was tired, {name2} showed",
        "when {name1} seemed worried, {name2} showed",
        "when {name1} looked nervous, {name2} showed",
        "when {name1} felt nervous, {name2} showed",
        "when {name1} called, {name2} showed",
        "when {name1} was alone, {name2} showed",
        "when {name1} looked tired, {name2} showed",
        "when {name1} stumbled, {name2} showed",
        "when {name1} was interested, {name2} showed",
        "when {name1} looked cold, {name2} showed",
        "when {name1} felt hungry, {name2} showed",
        "when {name1} came back, {name2} showed",
        "when {name1} stood there, {name2} showed",
        "when {name1} was worried, {name2} showed",
        "when {name1} appeared, {name2} showed",
        "when {name1} was curious, {name2} showed",
        "when {name1} was cold, {name2} showed",
        "when {name1} looked happy, {name2} showed",
        "when {name1} sobbed, {name2} showed",
        "when {name1} felt sick, {name2} showed",
        "when {name1} was nervous, {name2} showed",
        "when {name1} felt confused, {name2} showed",
        "when {name1} seemed sad, {name2} showed",
        "when {name1} felt angry, {name2} showed",
        "when {name1} looked confused, {name2} showed",
        "when {name1} entered, {name2} showed",
        "when {name1} arrived, {name2} showed",
        "when {name1} looked sick, {name2} showed",
        "when {name1} was happy, {name2} showed",
        "when {name1} slipped, {name2} showed",
        "when {name1} looked afraid, {name2} showed",
        "when {name1} was sad, {name2} showed",
        "when {name1} felt hurt, {name2} showed",
        "when {name1} looked hungry, {name2} showed",
        "when {name1} was learning, {name2} showed",
        "when {name1} hesitated, {name2} showed",
        "when {name1} felt upset, {name2} showed",
        "when {name1} shouted, {name2} showed",
        "when {name1} cried, {name2} showed",
        "when {name1} felt happy, {name2} showed",
        "when {name1} was confused, {name2} showed",
        "when {name1} looked hurt, {name2} showed",
        "when {name1} fell, {name2} showed",
        "when {name1} seemed angry, {name2} showed",
        "when {name1} was sick, {name2} showed",
        "when {name1} was hungry, {name2} showed",
        "when {name1} stopped, {name2} showed",
        "when {name1} was hurt, {name2} showed",
        "when {name1} was afraid, {name2} showed",
        "when {name1} looked angry, {name2} showed",
        "when {name1} was angry, {name2} showed",
        "when {name1} looked upset, {name2} showed",
        "when {name1} was upset, {name2} showed",
        "when {name1} needed help, {name2} showed",
        "when {name1} showed up, {name2} showed",
        "when {name1} tripped, {name2} showed",
        "when {name1} was stuck, {name2} showed",
        # beckoned (87 templates)
        "when {name1} looked afraid, {name2} beckoned",
        "when {name1} felt lost, {name2} beckoned",
        "when {name1} felt alone, {name2} beckoned",
        "when {name1} felt afraid, {name2} beckoned",
        "when {name1} was alone, {name2} beckoned",
        "when {name1} was lost, {name2} beckoned",
        "when {name1} waited, {name2} beckoned",
        "when {name1} was afraid, {name2} beckoned",
        "when {name1} stood alone, {name2} beckoned",
        "when {name1} felt tired, {name2} beckoned",
        "when {name1} wondered, {name2} beckoned",
        "when {name1} felt confused, {name2} beckoned",
        "when {name1} felt worried, {name2} beckoned",
        "when {name1} looked alone, {name2} beckoned",
        "when {name1} felt cold, {name2} beckoned",
        "when {name1} sat alone, {name2} beckoned",
        "when {name1} looked sick, {name2} beckoned",
        "when {name1} looked angry, {name2} beckoned",
        "when {name1} felt angry, {name2} beckoned",
        "when {name1} looked lost, {name2} beckoned",
        "when {name1} looked scared, {name2} beckoned",
        "when {name1} was tired, {name2} beckoned",
        "when {name1} looked worried, {name2} beckoned",
        "when {name1} felt nervous, {name2} beckoned",
        "when {name1} was learning, {name2} beckoned",
        "when {name1} felt hurt, {name2} beckoned",
        "when {name1} was angry, {name2} beckoned",
        "when {name1} felt sick, {name2} beckoned",
        "when {name1} was sick, {name2} beckoned",
        "when {name1} seemed worried, {name2} beckoned",
        "when {name1} fell, {name2} beckoned",
        "when {name1} felt happy, {name2} beckoned",
        "when {name1} was hungry, {name2} beckoned",
        "when {name1} was worried, {name2} beckoned",
        "when {name1} was sad, {name2} beckoned",
        "when {name1} looked tired, {name2} beckoned",
        "when {name1} was scared, {name2} beckoned",
        "when {name1} felt scared, {name2} beckoned",
        "when {name1} slipped, {name2} beckoned",
        "when {name1} felt sad, {name2} beckoned",
        "when {name1} felt hungry, {name2} beckoned",
        "when {name1} stumbled, {name2} beckoned",
        "when {name1} was stuck, {name2} beckoned",
        "when {name1} was nervous, {name2} beckoned",
        "when {name1} looked hungry, {name2} beckoned",
        "when {name1} looked nervous, {name2} beckoned",
        "when {name1} walked in, {name2} beckoned",
        "when {name1} sobbed, {name2} beckoned",
        "when {name1} looked happy, {name2} beckoned",
        "when {name1} stood there, {name2} beckoned",
        "when {name1} seemed lost, {name2} beckoned",
        "when {name1} called, {name2} beckoned",
        "when {name1} looked sad, {name2} beckoned",
        "when {name1} seemed alone, {name2} beckoned",
        "when {name1} looked cold, {name2} beckoned",
        "when {name1} was happy, {name2} beckoned",
        "when {name1} arrived, {name2} beckoned",
        "when {name1} was hurt, {name2} beckoned",
        "when {name1} called out, {name2} beckoned",
        "when {name1} felt curious, {name2} beckoned",
        "when {name1} felt upset, {name2} beckoned",
        "when {name1} walked inside, {name2} beckoned",
        "when {name1} seemed angry, {name2} beckoned",
        "when {name1} seemed scared, {name2} beckoned",
        "when {name1} forgot, {name2} beckoned",
        "when {name1} looked bored, {name2} beckoned",
        "when {name1} entered, {name2} beckoned",
        "when {name1} came back, {name2} beckoned",
        "when {name1} felt bored, {name2} beckoned",
        "when {name1} was cold, {name2} beckoned",
        "when {name1} was interested, {name2} beckoned",
        "when {name1} looked hurt, {name2} beckoned",
        "when {name1} returned, {name2} beckoned",
        "when {name1} showed up, {name2} beckoned",
        "when {name1} needed help, {name2} beckoned",
        "when {name1} was confused, {name2} beckoned",
        "when {name1} was bored, {name2} beckoned",
        "when {name1} looked confused, {name2} beckoned",
        "when {name1} asked, {name2} beckoned",
        "when {name1} appeared, {name2} beckoned",
        "when {name1} cried, {name2} beckoned",
        "when {name1} stopped, {name2} beckoned",
        "when {name1} came home, {name2} beckoned",
        "when {name1} was curious, {name2} beckoned",
        "when {name1} seemed bored, {name2} beckoned",
        "when {name1} shouted, {name2} beckoned",
        "when {name1} tripped, {name2} beckoned",
        # followed (83 templates)
        "when {name1} felt bored, {name2} followed",
        "when {name1} sat alone, {name2} followed",
        "when {name1} looked bored, {name2} followed",
        "when {name1} looked alone, {name2} followed",
        "when {name1} felt tired, {name2} followed",
        "when {name1} felt alone, {name2} followed",
        "when {name1} felt worried, {name2} followed",
        "when {name1} looked afraid, {name2} followed",
        "when {name1} looked happy, {name2} followed",
        "when {name1} felt afraid, {name2} followed",
        "when {name1} was alone, {name2} followed",
        "when {name1} was interested, {name2} followed",
        "when {name1} was bored, {name2} followed",
        "when {name1} felt angry, {name2} followed",
        "when {name1} called out, {name2} followed",
        "when {name1} looked angry, {name2} followed",
        "when {name1} felt upset, {name2} followed",
        "when {name1} was hungry, {name2} followed",
        "when {name1} was tired, {name2} followed",
        "when {name1} looked lost, {name2} followed",
        "when {name1} looked scared, {name2} followed",
        "when {name1} felt hungry, {name2} followed",
        "when {name1} looked worried, {name2} followed",
        "when {name1} sobbed, {name2} followed",
        "when {name1} looked cold, {name2} followed",
        "when {name1} looked hungry, {name2} followed",
        "when {name1} looked tired, {name2} followed",
        "when {name1} felt sad, {name2} followed",
        "when {name1} felt scared, {name2} followed",
        "when {name1} felt cold, {name2} followed",
        "when {name1} was angry, {name2} followed",
        "when {name1} felt happy, {name2} followed",
        "when {name1} looked sick, {name2} followed",
        "when {name1} showed up, {name2} followed",
        "when {name1} was afraid, {name2} followed",
        "when {name1} looked upset, {name2} followed",
        "when {name1} was happy, {name2} followed",
        "when {name1} felt confused, {name2} followed",
        "when {name1} looked sad, {name2} followed",
        "when {name1} felt hurt, {name2} followed",
        "when {name1} called, {name2} followed",
        "when {name1} felt lost, {name2} followed",
        "when {name1} was learning, {name2} followed",
        "when {name1} fell, {name2} followed",
        "when {name1} seemed bored, {name2} followed",
        "when {name1} appeared, {name2} followed",
        "when {name1} walked in, {name2} followed",
        "when {name1} felt sick, {name2} followed",
        "when {name1} asked, {name2} followed",
        "when {name1} was sick, {name2} followed",
        "when {name1} walked inside, {name2} followed",
        "when {name1} looked nervous, {name2} followed",
        "when {name1} stood alone, {name2} followed",
        "when {name1} was sad, {name2} followed",
        "when {name1} waited, {name2} followed",
        "when {name1} looked confused, {name2} followed",
        "when {name1} was hurt, {name2} followed",
        "when {name1} came back, {name2} followed",
        "when {name1} seemed worried, {name2} followed",
        "when {name1} was scared, {name2} followed",
        "when {name1} looked hurt, {name2} followed",
        "when {name1} entered, {name2} followed",
        "when {name1} was worried, {name2} followed",
        "when {name1} seemed angry, {name2} followed",
        "when {name1} was confused, {name2} followed",
        "when {name1} cried, {name2} followed",
        "when {name1} seemed alone, {name2} followed",
        "when {name1} came inside, {name2} followed",
        "when {name1} was cold, {name2} followed",
        "when {name1} was upset, {name2} followed",
        "when {name1} seemed scared, {name2} followed",
        "when {name1} felt nervous, {name2} followed",
        "when {name1} was lost, {name2} followed",
        "when {name1} slipped, {name2} followed",
        "when {name1} wondered, {name2} followed",
        "when {name1} was nervous, {name2} followed",
        "when {name1} was stuck, {name2} followed",
        "when {name1} shouted, {name2} followed",
        "when {name1} stopped, {name2} followed",
        "when {name1} seemed confused, {name2} followed",
        "when {name1} arrived, {name2} followed",
        "when {name1} seemed sad, {name2} followed",
        "when {name1} stood there, {name2} followed",
        # urged (80 templates)
        "when {name1} seemed alone, {name2} urged",
        "when {name1} walked in, {name2} urged",
        "when {name1} sat alone, {name2} urged",
        "when {name1} seemed worried, {name2} urged",
        "when {name1} felt bored, {name2} urged",
        "when {name1} seemed lost, {name2} urged",
        "when {name1} walked inside, {name2} urged",
        "when {name1} stood alone, {name2} urged",
        "when {name1} looked afraid, {name2} urged",
        "when {name1} felt hurt, {name2} urged",
        "when {name1} looked alone, {name2} urged",
        "when {name1} looked tired, {name2} urged",
        "when {name1} wondered, {name2} urged",
        "when {name1} looked hurt, {name2} urged",
        "when {name1} felt cold, {name2} urged",
        "when {name1} felt alone, {name2} urged",
        "when {name1} was learning, {name2} urged",
        "when {name1} was angry, {name2} urged",
        "when {name1} looked angry, {name2} urged",
        "when {name1} looked cold, {name2} urged",
        "when {name1} looked happy, {name2} urged",
        "when {name1} was alone, {name2} urged",
        "when {name1} was lost, {name2} urged",
        "when {name1} felt angry, {name2} urged",
        "when {name1} felt lost, {name2} urged",
        "when {name1} looked sick, {name2} urged",
        "when {name1} felt confused, {name2} urged",
        "when {name1} felt afraid, {name2} urged",
        "when {name1} was sad, {name2} urged",
        "when {name1} felt worried, {name2} urged",
        "when {name1} was interested, {name2} urged",
        "when {name1} felt curious, {name2} urged",
        "when {name1} sobbed, {name2} urged",
        "when {name1} stood there, {name2} urged",
        "when {name1} seemed bored, {name2} urged",
        "when {name1} felt tired, {name2} urged",
        "when {name1} waited, {name2} urged",
        "when {name1} looked confused, {name2} urged",
        "when {name1} was worried, {name2} urged",
        "when {name1} looked lost, {name2} urged",
        "when {name1} was bored, {name2} urged",
        "when {name1} looked worried, {name2} urged",
        "when {name1} cried, {name2} urged",
        "when {name1} was sick, {name2} urged",
        "when {name1} was confused, {name2} urged",
        "when {name1} was afraid, {name2} urged",
        "when {name1} felt sick, {name2} urged",
        "when {name1} slipped, {name2} urged",
        "when {name1} looked bored, {name2} urged",
        "when {name1} came back, {name2} urged",
        "when {name1} needed help, {name2} urged",
        "when {name1} was hurt, {name2} urged",
        "when {name1} called, {name2} urged",
        "when {name1} felt sad, {name2} urged",
        "when {name1} felt happy, {name2} urged",
        "when {name1} fell, {name2} urged",
        "when {name1} looked upset, {name2} urged",
        "when {name1} felt upset, {name2} urged",
        "when {name1} returned, {name2} urged",
        "when {name1} was scared, {name2} urged",
        "when {name1} forgot, {name2} urged",
        "when {name1} was tired, {name2} urged",
        "when {name1} called out, {name2} urged",
        "when {name1} was cold, {name2} urged",
        "when {name1} was happy, {name2} urged",
        "when {name1} looked hungry, {name2} urged",
        "when {name1} felt scared, {name2} urged",
        "when {name1} looked scared, {name2} urged",
        "when {name1} was stuck, {name2} urged",
        "when {name1} felt hungry, {name2} urged",
        "when {name1} entered, {name2} urged",
        "when {name1} seemed scared, {name2} urged",
        "when {name1} was hungry, {name2} urged",
        "when {name1} looked sad, {name2} urged",
        "when {name1} looked nervous, {name2} urged",
        "when {name1} was curious, {name2} urged",
        "when {name1} showed up, {name2} urged",
        "when {name1} came home, {name2} urged",
        "when {name1} asked, {name2} urged",
        "when {name1} felt nervous, {name2} urged",
        # told (74 templates)
        "when {name1} looked lost, {name2} told",
        "when {name1} seemed bored, {name2} told",
        "when {name1} felt bored, {name2} told",
        "when {name1} was afraid, {name2} told",
        "when {name1} was lost, {name2} told",
        "when {name1} waited, {name2} told",
        "when {name1} was bored, {name2} told",
        "when {name1} wondered, {name2} told",
        "when {name1} felt lost, {name2} told",
        "when {name1} looked afraid, {name2} told",
        "when {name1} felt afraid, {name2} told",
        "when {name1} looked bored, {name2} told",
        "when {name1} looked hurt, {name2} told",
        "when {name1} looked worried, {name2} told",
        "when {name1} looked sick, {name2} told",
        "when {name1} walked in, {name2} told",
        "when {name1} seemed confused, {name2} told",
        "when {name1} stood alone, {name2} told",
        "when {name1} seemed lost, {name2} told",
        "when {name1} seemed worried, {name2} told",
        "when {name1} felt angry, {name2} told",
        "when {name1} felt worried, {name2} told",
        "when {name1} stepped inside, {name2} told",
        "when {name1} was tired, {name2} told",
        "when {name1} was scared, {name2} told",
        "when {name1} was worried, {name2} told",
        "when {name1} was hurt, {name2} told",
        "when {name1} looked tired, {name2} told",
        "when {name1} looked alone, {name2} told",
        "when {name1} looked angry, {name2} told",
        "when {name1} felt tired, {name2} told",
        "when {name1} felt cold, {name2} told",
        "when {name1} looked cold, {name2} told",
        "when {name1} seemed scared, {name2} told",
        "when {name1} was hungry, {name2} told",
        "when {name1} was angry, {name2} told",
        "when {name1} stumbled, {name2} told",
        "when {name1} felt confused, {name2} told",
        "when {name1} was cold, {name2} told",
        "when {name1} hesitated, {name2} told",
        "when {name1} seemed alone, {name2} told",
        "when {name1} looked scared, {name2} told",
        "when {name1} felt scared, {name2} told",
        "when {name1} felt hurt, {name2} told",
        "when {name1} was interested, {name2} told",
        "when {name1} seemed angry, {name2} told",
        "when {name1} was alone, {name2} told",
        "when {name1} felt hungry, {name2} told",
        "when {name1} looked upset, {name2} told",
        "when {name1} asked, {name2} told",
        "when {name1} looked hungry, {name2} told",
        "when {name1} felt sick, {name2} told",
        "when {name1} arrived, {name2} told",
        "when {name1} was learning, {name2} told",
        "when {name1} was sick, {name2} told",
        "when {name1} entered, {name2} told",
        "when {name1} looked confused, {name2} told",
        "when {name1} felt upset, {name2} told",
        "when {name1} seemed sad, {name2} told",
        "when {name1} sobbed, {name2} told",
        "when {name1} was upset, {name2} told",
        "when {name1} cried, {name2} told",
        "when {name1} was sad, {name2} told",
        "when {name1} slipped, {name2} told",
        "when {name1} was curious, {name2} told",
        "when {name1} was stuck, {name2} told",
        "when {name1} felt alone, {name2} told",
        "when {name1} felt curious, {name2} told",
        "when {name1} called, {name2} told",
        "when {name1} felt sad, {name2} told",
        "when {name1} called out, {name2} told",
        "when {name1} needed help, {name2} told",
        "when {name1} was nervous, {name2} told",
        "when {name1} felt nervous, {name2} told",
        # supported (73 templates)
        "when {name1} looked bored, {name2} supported",
        "when {name1} seemed bored, {name2} supported",
        "when {name1} felt bored, {name2} supported",
        "when {name1} entered, {name2} supported",
        "when {name1} was bored, {name2} supported",
        "when {name1} felt cold, {name2} supported",
        "when {name1} looked alone, {name2} supported",
        "when {name1} looked nervous, {name2} supported",
        "when {name1} sobbed, {name2} supported",
        "when {name1} looked lost, {name2} supported",
        "when {name1} felt nervous, {name2} supported",
        "when {name1} looked angry, {name2} supported",
        "when {name1} stood alone, {name2} supported",
        "when {name1} looked tired, {name2} supported",
        "when {name1} looked afraid, {name2} supported",
        "when {name1} looked sick, {name2} supported",
        "when {name1} was lost, {name2} supported",
        "when {name1} seemed worried, {name2} supported",
        "when {name1} felt alone, {name2} supported",
        "when {name1} felt angry, {name2} supported",
        "when {name1} felt lost, {name2} supported",
        "when {name1} was nervous, {name2} supported",
        "when {name1} seemed lost, {name2} supported",
        "when {name1} looked cold, {name2} supported",
        "when {name1} was alone, {name2} supported",
        "when {name1} felt curious, {name2} supported",
        "when {name1} looked scared, {name2} supported",
        "when {name1} looked hungry, {name2} supported",
        "when {name1} felt tired, {name2} supported",
        "when {name1} seemed scared, {name2} supported",
        "when {name1} felt confused, {name2} supported",
        "when {name1} felt sad, {name2} supported",
        "when {name1} walked inside, {name2} supported",
        "when {name1} was angry, {name2} supported",
        "when {name1} was cold, {name2} supported",
        "when {name1} looked sad, {name2} supported",
        "when {name1} wondered, {name2} supported",
        "when {name1} was tired, {name2} supported",
        "when {name1} sat alone, {name2} supported",
        "when {name1} looked worried, {name2} supported",
        "when {name1} walked in, {name2} supported",
        "when {name1} waited, {name2} supported",
        "when {name1} felt worried, {name2} supported",
        "when {name1} felt sick, {name2} supported",
        "when {name1} was scared, {name2} supported",
        "when {name1} felt scared, {name2} supported",
        "when {name1} was worried, {name2} supported",
        "when {name1} arrived, {name2} supported",
        "when {name1} was interested, {name2} supported",
        "when {name1} felt hungry, {name2} supported",
        "when {name1} felt happy, {name2} supported",
        "when {name1} fell, {name2} supported",
        "when {name1} called out, {name2} supported",
        "when {name1} felt upset, {name2} supported",
        "when {name1} felt afraid, {name2} supported",
        "when {name1} called, {name2} supported",
        "when {name1} felt hurt, {name2} supported",
        "when {name1} appeared, {name2} supported",
        "when {name1} was sick, {name2} supported",
        "when {name1} was hungry, {name2} supported",
        "when {name1} was afraid, {name2} supported",
        "when {name1} looked happy, {name2} supported",
        "when {name1} cried, {name2} supported",
        "when {name1} was sad, {name2} supported",
        "when {name1} slipped, {name2} supported",
        "when {name1} showed up, {name2} supported",
        "when {name1} stumbled, {name2} supported",
        "when {name1} was happy, {name2} supported",
        "when {name1} shouted, {name2} supported",
        "when {name1} was learning, {name2} supported",
        "when {name1} returned, {name2} supported",
        "when {name1} needed help, {name2} supported",
        "when {name1} asked, {name2} supported",
        # hugged (56 templates)
        "when {name1} was interested, {name2} hugged",
        "when {name1} looked cold, {name2} hugged",
        "when {name1} looked confused, {name2} hugged",
        "when {name1} seemed bored, {name2} hugged",
        "when {name1} looked nervous, {name2} hugged",
        "when {name1} looked scared, {name2} hugged",
        "when {name1} felt bored, {name2} hugged",
        "when {name1} felt cold, {name2} hugged",
        "when {name1} looked bored, {name2} hugged",
        "when {name1} felt lost, {name2} hugged",
        "when {name1} was cold, {name2} hugged",
        "when {name1} looked afraid, {name2} hugged",
        "when {name1} looked lost, {name2} hugged",
        "when {name1} looked worried, {name2} hugged",
        "when {name1} felt hurt, {name2} hugged",
        "when {name1} was bored, {name2} hugged",
        "when {name1} felt angry, {name2} hugged",
        "when {name1} slipped, {name2} hugged",
        "when {name1} felt hungry, {name2} hugged",
        "when {name1} looked hungry, {name2} hugged",
        "when {name1} walked in, {name2} hugged",
        "when {name1} felt sick, {name2} hugged",
        "when {name1} shouted, {name2} hugged",
        "when {name1} felt afraid, {name2} hugged",
        "when {name1} called, {name2} hugged",
        "when {name1} was scared, {name2} hugged",
        "when {name1} felt scared, {name2} hugged",
        "when {name1} seemed angry, {name2} hugged",
        "when {name1} was hungry, {name2} hugged",
        "when {name1} was lost, {name2} hugged",
        "when {name1} looked sad, {name2} hugged",
        "when {name1} seemed worried, {name2} hugged",
        "when {name1} looked angry, {name2} hugged",
        "when {name1} looked tired, {name2} hugged",
        "when {name1} looked alone, {name2} hugged",
        "when {name1} was afraid, {name2} hugged",
        "when {name1} felt worried, {name2} hugged",
        "when {name1} felt alone, {name2} hugged",
        "when {name1} was worried, {name2} hugged",
        "when {name1} was angry, {name2} hugged",
        "when {name1} waited, {name2} hugged",
        "when {name1} asked, {name2} hugged",
        "when {name1} cried, {name2} hugged",
        "when {name1} felt curious, {name2} hugged",
        "when {name1} looked hurt, {name2} hugged",
        "when {name1} was alone, {name2} hugged",
        "when {name1} was upset, {name2} hugged",
        "when {name1} felt upset, {name2} hugged",
        "when {name1} was hurt, {name2} hugged",
        "when {name1} stopped, {name2} hugged",
        "when {name1} felt tired, {name2} hugged",
        "when {name1} stumbled, {name2} hugged",
        "when {name1} fell, {name2} hugged",
        "when {name1} needed help, {name2} hugged",
        "when {name1} tripped, {name2} hugged",
        "when {name1} returned, {name2} hugged",
        # trusted (54 templates)
        "when {name1} stepped in, {name2} trusted",
        "when {name1} walked in, {name2} trusted",
        "when {name1} entered, {name2} trusted",
        "when {name1} seemed bored, {name2} trusted",
        "when {name1} seemed lost, {name2} trusted",
        "when {name1} looked lost, {name2} trusted",
        "when {name1} walked inside, {name2} trusted",
        "when {name1} felt lost, {name2} trusted",
        "when {name1} was cold, {name2} trusted",
        "when {name1} wondered, {name2} trusted",
        "when {name1} felt hungry, {name2} trusted",
        "when {name1} looked bored, {name2} trusted",
        "when {name1} looked alone, {name2} trusted",
        "when {name1} looked confused, {name2} trusted",
        "when {name1} waited, {name2} trusted",
        "when {name1} was interested, {name2} trusted",
        "when {name1} was learning, {name2} trusted",
        "when {name1} felt cold, {name2} trusted",
        "when {name1} seemed scared, {name2} trusted",
        "when {name1} felt bored, {name2} trusted",
        "when {name1} was nervous, {name2} trusted",
        "when {name1} looked cold, {name2} trusted",
        "when {name1} was tired, {name2} trusted",
        "when {name1} looked upset, {name2} trusted",
        "when {name1} sobbed, {name2} trusted",
        "when {name1} was bored, {name2} trusted",
        "when {name1} looked scared, {name2} trusted",
        "when {name1} looked nervous, {name2} trusted",
        "when {name1} was hungry, {name2} trusted",
        "when {name1} felt nervous, {name2} trusted",
        "when {name1} was sick, {name2} trusted",
        "when {name1} looked sad, {name2} trusted",
        "when {name1} slipped, {name2} trusted",
        "when {name1} looked afraid, {name2} trusted",
        "when {name1} stumbled, {name2} trusted",
        "when {name1} called out, {name2} trusted",
        "when {name1} felt sick, {name2} trusted",
        "when {name1} was scared, {name2} trusted",
        "when {name1} arrived, {name2} trusted",
        "when {name1} seemed sad, {name2} trusted",
        "when {name1} seemed worried, {name2} trusted",
        "when {name1} looked happy, {name2} trusted",
        "when {name1} looked hungry, {name2} trusted",
        "when {name1} felt angry, {name2} trusted",
        "when {name1} was upset, {name2} trusted",
        "when {name1} felt afraid, {name2} trusted",
        "when {name1} called, {name2} trusted",
        "when {name1} was afraid, {name2} trusted",
        "when {name1} felt scared, {name2} trusted",
        "when {name1} was worried, {name2} trusted",
        "when {name1} looked worried, {name2} trusted",
        "when {name1} showed up, {name2} trusted",
        "when {name1} felt upset, {name2} trusted",
        "when {name1} was hurt, {name2} trusted",
        # helped (47 templates)
        "when {name1} looked bored, {name2} helped",
        "when {name1} was bored, {name2} helped",
        "when {name1} felt bored, {name2} helped",
        "when {name1} walked in, {name2} helped",
        "when {name1} called out, {name2} helped",
        "when {name1} looked alone, {name2} helped",
        "when {name1} walked inside, {name2} helped",
        "when {name1} looked tired, {name2} helped",
        "when {name1} seemed bored, {name2} helped",
        "when {name1} sobbed, {name2} helped",
        "when {name1} stood alone, {name2} helped",
        "when {name1} looked hungry, {name2} helped",
        "when {name1} waited, {name2} helped",
        "when {name1} felt nervous, {name2} helped",
        "when {name1} stepped in, {name2} helped",
        "when {name1} was hungry, {name2} helped",
        "when {name1} was interested, {name2} helped",
        "when {name1} looked sick, {name2} helped",
        "when {name1} looked cold, {name2} helped",
        "when {name1} was alone, {name2} helped",
        "when {name1} called, {name2} helped",
        "when {name1} seemed scared, {name2} helped",
        "when {name1} looked afraid, {name2} helped",
        "when {name1} was scared, {name2} helped",
        "when {name1} was upset, {name2} helped",
        "when {name1} felt cold, {name2} helped",
        "when {name1} was afraid, {name2} helped",
        "when {name1} looked scared, {name2} helped",
        "when {name1} was cold, {name2} helped",
        "when {name1} looked lost, {name2} helped",
        "when {name1} seemed sad, {name2} helped",
        "when {name1} stood there, {name2} helped",
        "when {name1} looked upset, {name2} helped",
        "when {name1} asked, {name2} helped",
        "when {name1} was tired, {name2} helped",
        "when {name1} felt tired, {name2} helped",
        "when {name1} looked sad, {name2} helped",
        "when {name1} seemed alone, {name2} helped",
        "when {name1} was learning, {name2} helped",
        "when {name1} seemed lost, {name2} helped",
        "when {name1} felt upset, {name2} helped",
        "when {name1} was sad, {name2} helped",
        "when {name1} looked nervous, {name2} helped",
        "when {name1} felt afraid, {name2} helped",
        "when {name1} sat alone, {name2} helped",
        "when {name1} was sick, {name2} helped",
        "when {name1} wondered, {name2} helped",
        # invited (40 templates)
        "when {name1} looked bored, {name2} invited",
        "when {name1} looked confused, {name2} invited",
        "when {name1} walked inside, {name2} invited",
        "when {name1} walked in, {name2} invited",
        "when {name1} looked tired, {name2} invited",
        "when {name1} felt lost, {name2} invited",
        "when {name1} asked, {name2} invited",
        "when {name1} felt tired, {name2} invited",
        "when {name1} looked scared, {name2} invited",
        "when {name1} felt bored, {name2} invited",
        "when {name1} looked alone, {name2} invited",
        "when {name1} seemed bored, {name2} invited",
        "when {name1} looked lost, {name2} invited",
        "when {name1} felt confused, {name2} invited",
        "when {name1} seemed lost, {name2} invited",
        "when {name1} wondered, {name2} invited",
        "when {name1} felt cold, {name2} invited",
        "when {name1} looked afraid, {name2} invited",
        "when {name1} stumbled, {name2} invited",
        "when {name1} looked angry, {name2} invited",
        "when {name1} felt afraid, {name2} invited",
        "when {name1} looked hungry, {name2} invited",
        "when {name1} felt alone, {name2} invited",
        "when {name1} called, {name2} invited",
        "when {name1} was tired, {name2} invited",
        "when {name1} felt hurt, {name2} invited",
        "when {name1} felt angry, {name2} invited",
        "when {name1} felt hungry, {name2} invited",
        "when {name1} was alone, {name2} invited",
        "when {name1} waited, {name2} invited",
        "when {name1} looked cold, {name2} invited",
        "when {name1} felt upset, {name2} invited",
        "when {name1} entered, {name2} invited",
        "when {name1} was hungry, {name2} invited",
        "when {name1} was sick, {name2} invited",
        "when {name1} sobbed, {name2} invited",
        "when {name1} called out, {name2} invited",
        "when {name1} felt worried, {name2} invited",
        "when {name1} was angry, {name2} invited",
        "when {name1} shouted, {name2} invited",
        # comforted (22 templates)
        "when {name1} entered, {name2} comforted",
        "when {name1} sobbed, {name2} comforted",
        "when {name1} stood alone, {name2} comforted",
        "when {name1} was worried, {name2} comforted",
        "when {name1} looked lost, {name2} comforted",
        "when {name1} wondered, {name2} comforted",
        "when {name1} looked bored, {name2} comforted",
        "when {name1} waited, {name2} comforted",
        "when {name1} fell, {name2} comforted",
        "when {name1} sat alone, {name2} comforted",
        "when {name1} arrived, {name2} comforted",
        "when {name1} stumbled, {name2} comforted",
        "when {name1} felt bored, {name2} comforted",
        "when {name1} looked hungry, {name2} comforted",
        "when {name1} was bored, {name2} comforted",
        "when {name1} looked afraid, {name2} comforted",
        "when {name1} was hungry, {name2} comforted",
        "when {name1} felt sick, {name2} comforted",
        "when {name1} felt afraid, {name2} comforted",
        "when {name1} shouted, {name2} comforted",
        "when {name1} felt worried, {name2} comforted",
        "when {name1} called, {name2} comforted",
        # led (13 templates)
        "when {name1} waited, {name2} led",
        "when {name1} walked inside, {name2} led",
        "when {name1} felt sad, {name2} led",
        "when {name1} felt tired, {name2} led",
        "when {name1} felt lost, {name2} led",
        "when {name1} looked alone, {name2} led",
        "when {name1} looked bored, {name2} led",
        "when {name1} felt alone, {name2} led",
        "when {name1} seemed worried, {name2} led",
        "when {name1} felt afraid, {name2} led",
        "when {name1} felt curious, {name2} led",
        "when {name1} felt hurt, {name2} led",
        "when {name1} cried, {name2} led",
        # thanked (10 templates)
        "when {name1} looked bored, {name2} thanked",
        "when {name1} looked confused, {name2} thanked",
        "when {name1} looked tired, {name2} thanked",
        "when {name1} looked worried, {name2} thanked",
        "when {name1} walked in, {name2} thanked",
        "when {name1} looked hungry, {name2} thanked",
        "when {name1} stepped in, {name2} thanked",
        "when {name1} looked lost, {name2} thanked",
        "when {name1} called out, {name2} thanked",
        "when {name1} asked, {name2} thanked",
    ]
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed)
        self.split = split
        # Pre-compute pronoun tokens
        him_ids = self.tokenizer.encode(" him", add_special_tokens=False)
        her_ids = self.tokenizer.encode(" her", add_special_tokens=False)
        self.him_token = him_ids[0] if him_ids else self.tokenizer.unk_token_id or 0
        self.her_token = her_ids[0] if her_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"ioi_relaxed_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Randomly decide if name1 (the one pronoun refers to) is male or female
        name1_is_male = self.rng.random() > 0.5
        
        if name1_is_male:
            # name1 is male, name2 is female
            name1 = self.rng.choice(self.MALE_NAMES)
            name2 = self.rng.choice(self.FEMALE_NAMES)
            correct_token = self.him_token
            incorrect_token = self.her_token
        else:
            # name1 is female, name2 is male
            name1 = self.rng.choice(self.FEMALE_NAMES)
            name2 = self.rng.choice(self.MALE_NAMES)
            correct_token = self.her_token
            incorrect_token = self.him_token
        
        # Build the sentence
        template = self.rng.choice(self.TEMPLATES)
        context = template.format(name1=name1, name2=name2)
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class IOIMixedTask(BinaryTask):
    """
    Indirect Object Identification (IOI) task - MIXED gender pairs.
    
    Same structure and thresholds as IOIRelaxedTask:
    - P(correct) > 0.3
    - Binary P(correct | {him, her}) > 0.8
    
    But allows ALL gender combinations:
    - Male name1, Female name2 (opposite)
    - Female name1, Male name2 (opposite)
    - Male name1, Male name2 (same)
    - Female name1, Female name2 (same)
    
    The target pronoun always refers to name1 (first clause subject).
    Same-gender pairs are harder because the model must use sentence
    structure rather than gender to identify the referent.
    
    Uses same 1394 templates across 20 verbs as IOIRelaxedTask.
    """
    
    MALE_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter"]
    FEMALE_NAMES = ["Mia", "Kim", "Rita", "Lily", "Maria"]
    
    # Same templates as IOIRelaxedTask
    TEMPLATES = IOIRelaxedTask.TEMPLATES
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed)
        self.split = split
        # Pre-compute pronoun tokens
        him_ids = self.tokenizer.encode(" him", add_special_tokens=False)
        her_ids = self.tokenizer.encode(" her", add_special_tokens=False)
        self.him_token = him_ids[0] if him_ids else self.tokenizer.unk_token_id or 0
        self.her_token = her_ids[0] if her_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"ioi_mixed_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Randomly decide if name1 is male or female (determines target pronoun)
        name1_is_male = self.rng.random() > 0.5
        
        # Randomly decide if name2 is same or different gender
        # 50% same gender, 50% different gender
        same_gender = self.rng.random() > 0.5
        
        if name1_is_male:
            name1 = self.rng.choice(self.MALE_NAMES)
            if same_gender:
                # Both male - pick different name
                available = [n for n in self.MALE_NAMES if n != name1]
                name2 = self.rng.choice(available)
            else:
                name2 = self.rng.choice(self.FEMALE_NAMES)
            correct_token = self.him_token
            incorrect_token = self.her_token
        else:
            name1 = self.rng.choice(self.FEMALE_NAMES)
            if same_gender:
                # Both female - pick different name
                available = [n for n in self.FEMALE_NAMES if n != name1]
                name2 = self.rng.choice(available)
            else:
                name2 = self.rng.choice(self.MALE_NAMES)
            correct_token = self.her_token
            incorrect_token = self.him_token
        
        # Build the sentence
        template = self.rng.choice(self.TEMPLATES)
        context = template.format(name1=name1, name2=name2)
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


class PronounDistractorTask(BinaryTask):
    """
    Pronoun prediction with distractor name of opposite gender.
    
    Task: First sentence has only ONE name (distractor). Second sentence introduces
    a name of the OPPOSITE gender and requires the matching possessive pronoun.
    
    Example: "when Rita was in the woods, Leo scratched his head."
    
    The distractor name (Rita) is of the opposite gender from the subject (Leo),
    testing whether the model correctly tracks which name the pronoun refers to.
    
    Uses possessive pronouns (his/her) with verbs like scratched, closed, rubbed, etc.
    """
    
    MALE_NAMES = ["Leo", "Alex", "Samuel", "Jose", "Peter"]
    FEMALE_NAMES = ["Mia", "Kim", "Rita", "Lily", "Maria"]
    
    # First sentence templates - one name only (the distractor)
    FIRST_SENTENCE_TEMPLATES = [
        "when {distractor} was in the woods,",
        "when {distractor} went to the park,",
        "when {distractor} walked home,",
        "while {distractor} sat by the fire,",
        "as {distractor} entered the room,",
        "after {distractor} came home,",
        "when {distractor} stood by the lake,",
        "while {distractor} played outside,",
    ]
    
    # Second sentence templates - verb + possessive pronoun
    # These verbs reliably predict possessive pronouns (his/her)
    # Tested to have high probability with the SimpleStories model
    SECOND_SENTENCE_TEMPLATES = [
        "{subject} scratched",   # scratched his/her (best: 83-98%)
        "{subject} closed",      # closed his/her (57-86%)
        "{subject} clapped",     # clapped his/her (40-62%)
        "{subject} rubbed",      # rubbed his/her (17-54%)
        "{subject} shook",       # shook his/her (26-33%)
        "{subject} raised",      # raised his/her (28-38%)
    ]
    
    def __init__(self, tokenizer: "PreTrainedTokenizer", seed: int = 42, split: str = "train"):
        super().__init__(tokenizer, seed)
        self.split = split
        # Pre-compute pronoun tokens (possessive: his/her)
        his_ids = self.tokenizer.encode(" his", add_special_tokens=False)
        her_ids = self.tokenizer.encode(" her", add_special_tokens=False)
        self.his_token = his_ids[0] if his_ids else self.tokenizer.unk_token_id or 0
        self.her_token = her_ids[0] if her_ids else self.tokenizer.unk_token_id or 0
    
    @property
    def name(self) -> str:
        return f"pronoun_distractor_{self.split}"
    
    def generate_example(self) -> TaskExample:
        # Randomly decide if subject (in second sentence) is male or female
        subject_is_male = self.rng.random() > 0.5
        
        if subject_is_male:
            # Subject is male, distractor is female
            subject = self.rng.choice(self.MALE_NAMES)
            distractor = self.rng.choice(self.FEMALE_NAMES)
            correct_token = self.his_token
            incorrect_token = self.her_token
        else:
            # Subject is female, distractor is male
            subject = self.rng.choice(self.FEMALE_NAMES)
            distractor = self.rng.choice(self.MALE_NAMES)
            correct_token = self.her_token
            incorrect_token = self.his_token
        
        # Build the sentence
        first_sentence = self.rng.choice(self.FIRST_SENTENCE_TEMPLATES).format(
            distractor=distractor
        )
        second_sentence = self.rng.choice(self.SECOND_SENTENCE_TEMPLATES).format(
            subject=subject
        )
        
        context = first_sentence + " " + second_sentence
        
        # Tokenize
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


# Registry of available tasks
TASK_REGISTRY = {
    "dummy_quote": DummyQuoteTask,
    "dummy_article": DummyArticleTask,
    "dummy_pronoun": DummyPronounTask,
    "dummy_pronoun_wrong": DummyPronounWrongTask,
    "dummy_pronoun_when": DummyPronounWhenTask,
    "dummy_pronoun_is": DummyPronounIsTask,
    "dummy_pronoun_evil": DummyPronounEvilTask,
    "dummy_pronoun_water": DummyPronounWaterTask,
    "dummy_pronoun_iswhen": DummyPronounIsWhenTask,
    "dummy_tense": DummyTenseTask,
    "ioi_strict": IOIStrictTask,
    "ioi_relaxed": IOIRelaxedTask,
    "ioi_mixed": IOIMixedTask,
    "pronoun_distractor": PronounDistractorTask,
}


def get_task(task_name: str, tokenizer: PreTrainedTokenizer, seed: int = 42, split: str = "train") -> BinaryTask:
    """
    Get a task by name.
    
    Args:
        task_name: Name of the task (e.g., "dummy_pronoun")
        tokenizer: Tokenizer to use
        seed: Random seed
        split: "train" or "val" - for tasks that support train/val splits
        
    Returns:
        BinaryTask instance
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    
    task_class = TASK_REGISTRY[task_name]
    
    # Check if task supports split parameter
    import inspect
    sig = inspect.signature(task_class.__init__)
    if 'split' in sig.parameters:
        return task_class(tokenizer, seed=seed, split=split)
    else:
        return task_class(tokenizer, seed=seed)


class TaskDataset:
    """
    Wraps a BinaryTask as an iterable dataset for dashboard computation.
    
    Yields batches of token sequences from the task's positive examples.
    Can be used with the visualization module's dashboard computation.
    
    Usage:
        task = get_task("dummy_quote", tokenizer)
        dataset = TaskDataset(task, n_samples=1000, max_length=256)
        
        for batch in dataset:
            # batch is a tensor of shape [batch_size, max_length]
            ...
    """
    
    def __init__(
        self,
        task: BinaryTask,
        n_samples: int = 1000,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self.task = task
        self.n_samples = n_samples
        self.max_length = max_length
        self.batch_size = batch_size
    
    def __iter__(self):
        """Iterate over batches of task examples."""
        generated = 0
        while generated < self.n_samples:
            batch_size = min(self.batch_size, self.n_samples - generated)
            positive_ids, _, _, _, _ = self.task.generate_batch(batch_size, self.max_length)
            yield positive_ids
            generated += batch_size
    
    def __len__(self):
        """Number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def get_all_texts(self, n_samples: Optional[int] = None) -> List[str]:
        """
        Generate text examples from the task for dashboard computation.
        
        Returns list of decoded text strings from positive examples.
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        texts = []
        generated = 0
        
        while generated < n_samples:
            batch_size = min(self.batch_size, n_samples - generated)
            positive_ids, _, _, _, _ = self.task.generate_batch(batch_size, self.max_length)
            
            for i in range(positive_ids.shape[0]):
                # Decode tokens to text
                tokens = positive_ids[i].tolist()
                # Remove padding
                pad_id = self.task.tokenizer.pad_token_id or 0
                tokens = [t for t in tokens if t != pad_id]
                text = self.task.tokenizer.decode(tokens)
                texts.append(text)
            
            generated += batch_size
        
        return texts[:n_samples]

