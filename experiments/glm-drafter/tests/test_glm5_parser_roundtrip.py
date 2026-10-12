"""Parser round-trip tests against the real GLM-5.3-Flash tokenizer."""

import torch
from deepspec.data.parser import TEMPLATE_REGISTRY, GeneralParser
from transformers import PreTrainedTokenizerFast

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"


def _conversation() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": (THINK_OPEN + "Two plus two makes four." + THINK_CLOSE + "The answer is 4."),
        },
        {"role": "user", "content": "And 3 + 3?"},
        {
            "role": "assistant",
            "content": (
                THINK_OPEN + "Three plus three makes six." + THINK_CLOSE + "The answer is 6."
            ),
        },
    ]


def test_rendered_text_uses_ascii_think_tags(
    glm_tokenizer: PreTrainedTokenizerFast,
) -> None:
    text = glm_tokenizer.apply_chat_template(
        _conversation(), tokenize=False, add_generation_prompt=False
    )
    assert THINK_OPEN in text
    assert THINK_CLOSE in text


def test_parse_loss_masks_exclude_think_open(
    glm_tokenizer: PreTrainedTokenizerFast,
) -> None:
    template = TEMPLATE_REGISTRY.get("glm")
    parser = GeneralParser(tokenizer=glm_tokenizer, chat_template=template)
    out = parser.parse(_conversation(), max_length=512)
    loss_mask = out["loss_mask"]
    ids = out["input_ids"]
    decoded = glm_tokenizer.decode(ids)
    assert decoded == glm_tokenizer.apply_chat_template(
        _conversation(), tokenize=False, add_generation_prompt=False
    )
    assert loss_mask[0] == 0

    matches = list(torch.nonzero(loss_mask, as_tuple=False).flatten().tolist())
    assert matches, "expected a non-empty loss region"

    prefix_len = len(
        glm_tokenizer.encode(
            decoded.split(THINK_OPEN)[0] + THINK_OPEN,
            add_special_tokens=False,
        )
    )
    first_loss = matches[0]
    assert first_loss == prefix_len, (
        "loss must start after the think-open marker of the first assistant turn"
    )

    reasoning_ids = glm_tokenizer.encode(
        "Two plus two makes four." + THINK_CLOSE + "The answer is 4.",
        add_special_tokens=False,
    )
    span = ids[first_loss : first_loss + len(reasoning_ids)]
    assert span.tolist() == reasoning_ids
    assert loss_mask[first_loss : first_loss + len(reasoning_ids)].all()
