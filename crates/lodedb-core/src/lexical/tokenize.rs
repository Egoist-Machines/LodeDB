//! Dependency-free tokenizer matching `lodedb.engine._lexical.tokenize`.

/// Tokenizes text for lexical matching, preserving code-like interior separators.
///
/// A token is an ASCII alphanumeric run that may contain interior single `-`,
/// `.`, or `/` separators when the separator is followed by another ASCII
/// alphanumeric. Leading, trailing, and doubled separators are boundaries.
pub fn tokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];
        if is_ascii_token_char(ch) {
            current.push(ch);
            index += 1;
            continue;
        }
        if is_separator(ch)
            && !current.is_empty()
            && chars
                .get(index + 1)
                .is_some_and(|next| is_ascii_token_char(*next))
        {
            current.push(ch);
            index += 1;
            continue;
        }
        if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
        index += 1;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn is_ascii_token_char(ch: char) -> bool {
    ch.is_ascii_digit() || ch.is_ascii_lowercase()
}

fn is_separator(ch: char) -> bool {
    matches!(ch, '-' | '.' | '/')
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn preserves_code_like_shapes() {
        assert_eq!(
            tokenize("Error E1234 on 2024-01-15"),
            ["error", "e1234", "on", "2024-01-15"]
        );
        assert_eq!(tokenize("serial ABC-123-X"), ["serial", "abc-123-x"]);
        assert_eq!(
            tokenize("see v1.2.3 and a/b"),
            ["see", "v1.2.3", "and", "a/b"]
        );
    }

    #[test]
    fn doubled_or_boundary_separators_split() {
        assert_eq!(tokenize("a--b"), ["a", "b"]);
        assert_eq!(tokenize("-E1234-"), ["e1234"]);
        assert_eq!(tokenize("x... y"), ["x", "y"]);
    }
}
