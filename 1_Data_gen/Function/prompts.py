MMSTAR_PROMPT_TEMPLATE = (
    "Answer the following multiple-choice visual question.\n\n"
    "{question}\n\n"
    "Think carefully, but in the final response only output one line:\n"
    "Final answer: A, B, C, or D.\n"
)

OCRBENCH_PROMPT_TEMPLATE = (
    "You are given an image and a question about its visual / textual content. "
    "Read the image carefully and answer using only the text or content visible "
    "in the image.\n\n"
    "Question: {question}\n\n"
    "Think briefly if needed, but in the final response output ONE line:\n"
    "Final answer: <your concise answer>\n"
    "Do not wrap the answer in quotes or any extra commentary."
)
