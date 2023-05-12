def create_mask_text(text, offset, mask_len=1):
    prev_text = text[: offset]
    curr_text = text[offset : offset + mask_len]
    after_text = text[offset + mask_len:]
    return prev_text + "<mask>" + after_text + "\0" + curr_text

def create_mask_texts(text, mask_len=1):
    mask_texts = []
    offset = 0
    for idx in range(offset, len(text)-mask_len):
        mask_text = create_mask_text(text, idx, mask_len)
        mask_texts.append(mask_text)
    return mask_texts
