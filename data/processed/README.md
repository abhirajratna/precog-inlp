# Data Preprocessing Notes

## Changes Made

- Manually removed the initial and end headers
- Removed the Project Gutenberg license and Table of Contents

## Paragraph Segmentation

Performed paragraph segmentation to ensure the model is not trained on correlations between text length and authenticity (long text = human, short text = AI).

## Capitalization and Punctuation

Initially considered lowercasing all text, but decided against it because:
- Capitalization affects sentence structure and noun usage
- Removing punctuation would interfere with Task 1

## Illustration Removal

Removed numerous instances of `[Illustration]` tags from Pride and Prejudice. These included:

**Example 1:**
```
[Illustration: M^{r.} & M^{rs.} Bennet

[_Copyright 1894 by George Allen._]]
```

**Example 2:**
```
[Illustration:

     "He rode a black horse"
]
```

**Regex used:** `\[Illustration:.*?\]\]?`

## Italics Removal

Now gutenberg uses underscore to write in italics which I will have to remove as it is not part of Jane Austens writing
so I will remove them by `text = re.sub(r"_([^_]+)_", r"\1", text)` 
