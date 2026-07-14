from pathlib import Path

import pytest
from PIL import Image

from vlm.pair_features import SLOT_NAMES
from vlm.pair_prompt import (
    MARKED_PAIR_IDENTITY,
    PAIR_EVIDENCE_TOKENS,
    PAIR_INSTRUCTION_TEMPLATE,
    PAIR_SPECIAL_TOKENS,
    SOCIAL_RELATION_TOKEN,
    READOUT_QUESTIONS,
    TASK_DEFINITIONS,
    UNMARKED_PAIR_IDENTITY,
    add_pair_special_tokens,
    pair_special_token_ids,
    social_readout_prompt,
    task_conditioned_pair_instruction,
    task_conditioned_pair_prompt,
    validate_pair_prompt,
    validate_tokenized_pair_prompt,
)


class _FakeTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self.additional_special_tokens = ["<existing>"]
        self.vocab = {"<unk>": 0, "<existing>": 1}

    def add_special_tokens(self, spec, replace_additional_special_tokens=True):
        added = 0
        for token in spec["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        if replace_additional_special_tokens:
            self.additional_special_tokens = list(spec["additional_special_tokens"])
        else:
            for token in spec["additional_special_tokens"]:
                if token not in self.additional_special_tokens:
                    self.additional_special_tokens.append(token)
        return added

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        specials = sorted(self.vocab, key=len, reverse=True)
        ids = []
        index = 0
        while index < len(text):
            token = next((item for item in specials if text.startswith(item, index)), None)
            if token is not None:
                ids.append(self.vocab[token])
                index += len(token)
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids


def test_task_conditioning_changes_only_one_plain_text_field():
    assert tuple(token[1:-1] for token in PAIR_EVIDENCE_TOKENS) == SLOT_NAMES
    normalized = []
    for task, definition in TASK_DEFINITIONS.items():
        instruction = task_conditioned_pair_instruction(task)
        prompt = task_conditioned_pair_prompt(task)
        assert instruction == PAIR_INSTRUCTION_TEMPLATE.format(
            task_definition=definition,
            pair_identity=MARKED_PAIR_IDENTITY,
        )
        assert f"Task definition: {definition}" in prompt
        for token in PAIR_SPECIAL_TOKENS:
            assert prompt.count(token) == 1
        readout = social_readout_prompt(task)
        assert readout == (
            f"Question: {READOUT_QUESTIONS[task]} Answer (yes or no) : "
            f"{SOCIAL_RELATION_TOKEN}"
        )
        assert prompt.endswith(readout)
        assert prompt.endswith(SOCIAL_RELATION_TOKEN)
        assert "yes or no" in prompt.lower()      # readout now elicits a yes/no answer
        validate_pair_prompt(task)
        normalized.append(
            prompt.replace(definition, "{TASK_DEFINITION}").replace(readout, "{READOUT}")
        )
    assert len(set(normalized)) == 1


def test_unmarked_prompt_is_explicit_and_keeps_the_same_slot_schema():
    marked = task_conditioned_pair_prompt("lah")
    unmarked = task_conditioned_pair_prompt("lah", draw_bboxes=False)
    assert MARKED_PAIR_IDENTITY in marked and UNMARKED_PAIR_IDENTITY not in marked
    assert UNMARKED_PAIR_IDENTITY in unmarked and MARKED_PAIR_IDENTITY not in unmarked
    for token in PAIR_SPECIAL_TOKENS:
        assert marked.count(token) == unmarked.count(token) == 1
    validate_pair_prompt("lah", unmarked, draw_bboxes=False)


def test_special_registration_preserves_existing_tokens_and_is_idempotent():
    from vlm.pair_prompt import GRAPH_TOKENS
    total = len(PAIR_SPECIAL_TOKENS) + len(GRAPH_TOKENS)   # yes/no slots + generative gtoks
    tokenizer = _FakeTokenizer()
    assert add_pair_special_tokens(tokenizer) == total
    assert tokenizer.additional_special_tokens[0] == "<existing>"
    assert set(PAIR_SPECIAL_TOKENS).issubset(tokenizer.additional_special_tokens)
    assert set(GRAPH_TOKENS).issubset(tokenizer.additional_special_tokens)
    assert add_pair_special_tokens(tokenizer) == 0
    ids = pair_special_token_ids(tokenizer)
    assert len(ids) == total
    validate_tokenized_pair_prompt(tokenizer, task_conditioned_pair_prompt("lah"))


def test_prompt_validator_rejects_missing_or_duplicate_placeholders():
    prompt = task_conditioned_pair_prompt("lah")
    with pytest.raises(ValueError, match="once"):
        validate_pair_prompt("lah", prompt.replace("<person_a>", ""))
    with pytest.raises(ValueError, match="once"):
        validate_pair_prompt("lah", prompt + " <person_a>")
    with pytest.raises(ValueError, match="unknown social task"):
        task_conditioned_pair_prompt("other")


def test_local_qwen_chat_template_preserves_all_pair_tokens():
    transformers = pytest.importorskip("transformers")
    model_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    snapshots = sorted((model_root / "snapshots").glob("*"))
    if not snapshots:
        pytest.skip("local Qwen3-VL processor cache is unavailable")

    processor = transformers.AutoProcessor.from_pretrained(
        snapshots[-1], local_files_only=True
    )
    tokenizer = processor.tokenizer
    add_pair_special_tokens(tokenizer)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (32, 32), "black")},
                {"type": "text", "text": task_conditioned_pair_instruction("laeo")},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": social_readout_prompt("laeo")}],
        },
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    validate_tokenized_pair_prompt(tokenizer, rendered)
    positions = [rendered.index(token) for token in PAIR_SPECIAL_TOKENS]
    assert positions == sorted(positions)
    assert rendered.endswith(SOCIAL_RELATION_TOKEN)
    encoded = tokenizer.encode(rendered, add_special_tokens=False)
    assert encoded[-1] == pair_special_token_ids(tokenizer)[SOCIAL_RELATION_TOKEN]
