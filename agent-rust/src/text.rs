use std::borrow::Cow;

pub fn compact_whitespace(value: &str) -> Cow<'_, str> {
    let mut previous_was_space = true;
    let already_compact = value.chars().all(|character| {
        let is_space = character.is_whitespace();
        let valid = !is_space || (character == ' ' && !previous_was_space);
        previous_was_space = is_space;
        valid
    }) && !previous_was_space;
    if already_compact || value.is_empty() {
        return Cow::Borrowed(value);
    }

    let mut compact = String::with_capacity(value.len());
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_whitespace() {
            pending_space = !compact.is_empty();
        } else {
            if pending_space {
                compact.push(' ');
                pending_space = false;
            }
            compact.push(character);
        }
    }
    Cow::Owned(compact)
}

#[cfg(test)]
mod tests {
    use super::compact_whitespace;

    #[test]
    fn compacts_unicode_whitespace_without_edge_spaces() {
        assert_eq!(compact_whitespace(" \tone\u{2003}two\n "), "one two");
    }

    #[test]
    fn keeps_compact_text_borrowed() {
        assert!(matches!(
            compact_whitespace("already compact"),
            std::borrow::Cow::Borrowed(_)
        ));
    }
}
