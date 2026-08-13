use rust_latex_parser::{parse_equation, AccentKind, EqNode, MatrixKind};

const DISPLAY_MATH_DELIMITERS: [(&str, &str); 2] = [(r"\[", r"\]"), ("$$", "$$")];

pub(crate) fn message_lines(content: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut display_math: Option<(&str, &str, Vec<&str>)> = None;
    let mut code_fence: Option<char> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(marker) = code_fence {
            output.push(line.to_string());
            if is_code_fence(trimmed, marker) {
                code_fence = None;
            }
            continue;
        }

        if let Some(marker) = code_fence_marker(trimmed) {
            code_fence = Some(marker);
            output.push(line.to_string());
            continue;
        }

        if let Some((_, closing, source)) = display_math.as_mut() {
            if trimmed == *closing {
                output.push(render_latex(&source.join(" ")));
                display_math = None;
            } else {
                source.push(line);
            }
            continue;
        }

        if let Some(rendered) = render_single_line_display_math(trimmed) {
            output.push(rendered);
            continue;
        }

        if let Some((opening, closing)) = DISPLAY_MATH_DELIMITERS
            .iter()
            .find(|(opening, _)| trimmed == *opening)
        {
            display_math = Some((opening, closing, Vec::new()));
            continue;
        }

        output.push(render_inline_math(line));
    }

    if let Some((opening, _, source)) = display_math {
        output.push(opening.to_string());
        output.extend(source.into_iter().map(str::to_string));
    }

    output
}

fn code_fence_marker(text: &str) -> Option<char> {
    ['`', '~']
        .into_iter()
        .find(|marker| text.starts_with(&marker.to_string().repeat(3)))
}

fn is_code_fence(text: &str, marker: char) -> bool {
    text.starts_with(&marker.to_string().repeat(3))
}

fn render_single_line_display_math(text: &str) -> Option<String> {
    DISPLAY_MATH_DELIMITERS
        .iter()
        .find_map(|(opening, closing)| {
            text.strip_prefix(opening)
                .and_then(|content| content.strip_suffix(closing))
                .filter(|content| !content.is_empty())
                .map(render_latex)
        })
}

fn render_inline_math(line: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut cursor = 0;
    let mut in_code = false;

    while cursor < line.len() {
        let remaining = &line[cursor..];
        if remaining.starts_with('`') {
            in_code = !in_code;
            output.push('`');
            cursor += 1;
            continue;
        }

        if !in_code && remaining.starts_with(r"\(") {
            if let Some(end) = remaining[2..].find(r"\)") {
                output.push_str(&render_latex(&remaining[2..2 + end]));
                cursor += 2 + end + 2;
                continue;
            }
        }

        if !in_code && remaining.starts_with('$') && !remaining.starts_with("$$") {
            if let Some(end) = find_unescaped_dollar(&remaining[1..]) {
                output.push_str(&render_latex(&remaining[1..1 + end]));
                cursor += 1 + end + 1;
                continue;
            }
        }

        let character = remaining
            .chars()
            .next()
            .expect("cursor always points to a character boundary");
        output.push(character);
        cursor += character.len_utf8();
    }

    output
}

fn find_unescaped_dollar(text: &str) -> Option<usize> {
    text.char_indices().find_map(|(index, character)| {
        if character != '$' {
            return None;
        }
        let preceding_slashes = text[..index]
            .chars()
            .rev()
            .take_while(|character| *character == '\\')
            .count();
        preceding_slashes.is_multiple_of(2).then_some(index)
    })
}

fn render_latex(source: &str) -> String {
    render_node(&parse_equation(source.trim()))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_node(node: &EqNode) -> String {
    match node {
        EqNode::Text(text) | EqNode::TextBlock(text) => ascii_math_text(text),
        EqNode::Space(width) => {
            if *width > 0.0 {
                " ".to_string()
            } else {
                String::new()
            }
        }
        EqNode::Seq(nodes) => nodes.iter().map(render_node).collect(),
        EqNode::Sup(base, exponent) => {
            format!("{}^{}", render_node(base), render_script(exponent))
        }
        EqNode::Sub(base, subscript) => {
            format!("{}_{}", render_node(base), render_script(subscript))
        }
        EqNode::SupSub(base, exponent, subscript) => format!(
            "{}^{}_{}",
            render_node(base),
            render_script(exponent),
            render_script(subscript)
        ),
        EqNode::Frac(numerator, denominator) => format!(
            "{} / {}",
            render_fraction_part(numerator),
            render_fraction_part(denominator)
        ),
        EqNode::Sqrt(content) => format!("sqrt({})", render_node(content).trim()),
        EqNode::BigOp {
            symbol,
            lower,
            upper,
        } => {
            let mut rendered = ascii_math_text(symbol);
            if let Some(lower) = lower {
                rendered.push('_');
                rendered.push_str(&render_script(lower));
            }
            if let Some(upper) = upper {
                rendered.push('^');
                rendered.push_str(&render_script(upper));
            }
            rendered
        }
        EqNode::Accent(content, kind) => {
            format!("{}({})", accent_name(*kind), render_node(content).trim())
        }
        EqNode::Limit { name, lower } => lower.as_ref().map_or_else(
            || name.clone(),
            |lower| format!("{}_{}", name, render_script(lower)),
        ),
        EqNode::MathFont { content, .. } => render_node(content),
        EqNode::Delimited {
            left,
            right,
            content,
        } => format!(
            "{}{}{}",
            ascii_math_text(left),
            render_node(content).trim(),
            ascii_math_text(right)
        ),
        EqNode::Matrix { kind, rows } => render_matrix(*kind, rows),
        EqNode::Cases { rows } => {
            let rows = rows
                .iter()
                .map(|(value, condition)| {
                    let value = render_node(value);
                    condition.as_ref().map_or(value.clone(), |condition| {
                        format!("{} if {}", value.trim(), render_node(condition).trim())
                    })
                })
                .collect::<Vec<_>>()
                .join("; ");
            format!("{{ {rows} }}")
        }
        EqNode::Binom(top, bottom) => format!(
            "binom({}, {})",
            render_node(top).trim(),
            render_node(bottom).trim()
        ),
        EqNode::Brace {
            content,
            label,
            over,
        } => {
            let name = if *over { "overbrace" } else { "underbrace" };
            label.as_ref().map_or_else(
                || format!("{name}({})", render_node(content).trim()),
                |label| {
                    format!(
                        "{name}({}, {})",
                        render_node(content).trim(),
                        render_node(label).trim()
                    )
                },
            )
        }
        EqNode::StackRel {
            base,
            annotation,
            over,
        } => {
            let name = if *over { "overset" } else { "underset" };
            format!(
                "{name}({}, {})",
                render_node(annotation).trim(),
                render_node(base).trim()
            )
        }
    }
}

fn render_script(node: &EqNode) -> String {
    let rendered = render_node(node);
    let rendered = rendered.trim();
    if rendered
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
    {
        rendered.to_string()
    } else {
        format!("({rendered})")
    }
}

fn render_fraction_part(node: &EqNode) -> String {
    let rendered = render_node(node);
    let rendered = rendered.trim();
    let is_atomic = matches!(
        node,
        EqNode::Text(_)
            | EqNode::TextBlock(_)
            | EqNode::Sup(_, _)
            | EqNode::Sub(_, _)
            | EqNode::SupSub(_, _, _)
            | EqNode::Sqrt(_)
            | EqNode::Limit { .. }
            | EqNode::MathFont { .. }
            | EqNode::Delimited { .. }
    ) || matches!(node, EqNode::Seq(nodes) if nodes.iter().all(|node| {
        !matches!(node, EqNode::Space(width) if *width > 0.0)
    }));
    if is_atomic {
        rendered.to_string()
    } else {
        format!("({rendered})")
    }
}

fn accent_name(kind: AccentKind) -> &'static str {
    match kind {
        AccentKind::Hat => "hat",
        AccentKind::Bar => "bar",
        AccentKind::Dot => "dot",
        AccentKind::DoubleDot => "ddot",
        AccentKind::Tilde => "tilde",
        AccentKind::Vec => "vec",
    }
}

fn render_matrix(kind: MatrixKind, rows: &[Vec<EqNode>]) -> String {
    let (left, right) = match kind {
        MatrixKind::Plain => ("", ""),
        MatrixKind::Paren => ("(", ")"),
        MatrixKind::Bracket => ("[", "]"),
        MatrixKind::VBar => ("|", "|"),
        MatrixKind::DoubleVBar => ("||", "||"),
        MatrixKind::Brace => ("{", "}"),
    };
    let rows = rows
        .iter()
        .map(|row| row.iter().map(render_node).collect::<Vec<_>>().join(", "))
        .collect::<Vec<_>>()
        .join("; ");
    format!("{left}{rows}{right}")
}

fn ascii_math_text(text: &str) -> String {
    if text == r"\top" {
        return "T".to_string();
    }

    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        match ascii_math_symbol(character) {
            Some(symbol) => output.push_str(symbol),
            None => output.push(character),
        }
    }
    output
}

fn ascii_math_symbol(character: char) -> Option<&'static str> {
    Some(match character {
        '⊤' => "T",
        '√' => "sqrt",
        '∑' => "sum",
        '∏' => "prod",
        '∫' => "int",
        '∞' => "inf",
        '≤' => "<=",
        '≥' => ">=",
        '≠' => "!=",
        '≈' => "~=",
        '→' => "->",
        '←' => "<-",
        '↔' => "<->",
        '×' | '·' => "*",
        '÷' => "/",
        '±' => "+/-",
        '∓' => "-/+",
        '∈' => "in",
        '∉' => "not in",
        '∀' => "forall",
        '∃' => "exists",
        '∂' => "partial",
        '∇' => "nabla",
        '…' | '⋯' => "...",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_display_math_without_delimiter_rows() {
        let content = concat!(
            "The standard formula is:\n\n",
            "\\[\n",
            "\\text{Attention}(Q, K, V) = ",
            "\\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V\n",
            "\\]\n"
        );

        assert_eq!(
            message_lines(content),
            [
                "The standard formula is:",
                "",
                "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
            ]
        );
    }

    #[test]
    fn renders_inline_math_but_preserves_inline_code() {
        assert_eq!(
            message_lines(r"Use \(h_{i+1}=h_i+f_i(h_i)\), not `\frac{x}{y}`."),
            ["Use h_(i + 1) = h_i + f_i(h_i), not `\\frac{x}{y}`."]
        );
    }

    #[test]
    fn renders_multi_head_and_single_line_display_math() {
        assert_eq!(
            message_lines(
                r"$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$"
            ),
            ["MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O"]
        );
    }

    #[test]
    fn preserves_math_delimiters_inside_code_fences() {
        assert_eq!(
            message_lines("```latex\n\\[\n\\frac{x}{y}\n\\]\n```"),
            ["```latex", "\\[", "\\frac{x}{y}", "\\]", "```"]
        );
    }

    #[test]
    fn leaves_unclosed_math_delimiters_unchanged() {
        assert_eq!(
            message_lines("before\n\\[\nx + 1"),
            ["before", "\\[", "x + 1"]
        );
        assert_eq!(message_lines("cost is $5"), ["cost is $5"]);
    }
}
