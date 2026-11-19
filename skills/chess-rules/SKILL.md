---
name: chess-rules
description: Select one legal move in standard chess from the position supplied by the evaluation harness, and return it in strict UCI notation.
---

# Chess move selection

Play the side to move in the supplied position. Choose a move using your own reasoning; do not use a chess engine, external analysis, or another player. The harness handles move submission and follows the server's clock and game-result decisions.

## Read the position

- FEN has six fields: piece placement, active color (`w` or `b`), castling rights, en passant target, halfmove clock, and fullmove number.
- Placement lists ranks 8 through 1; each rank runs from file a through h. Digits count empty squares. Uppercase pieces are White and lowercase pieces are Black: `K/Q/R/B/N/P` mean king/queen/rook/bishop/knight/pawn. White pawns advance toward rank 8; Black pawns advance toward rank 1.
- Use the current FEN and supplied move history. Castling rights and en passant availability cannot be inferred from piece locations alone. If a legal-move list is provided, select from it; otherwise determine legality yourself.

## Apply standard rules

- Move exactly one piece of the active color. Do not capture your own pieces. Bishops move diagonally, rooks along ranks or files, and queens in either way. These pieces cannot jump over occupied squares. Knights move two squares on one axis and one on the other and may jump.
- Pawns advance one square into an empty square. From their starting rank they may advance two squares only if both squares are empty. Pawns normally capture one square diagonally forward, not straight ahead. On reaching the last rank they must promote to a queen, rook, bishop, or knight, including when capturing.
- Kings move one square in any direction. A move must never leave your own king in check; pinned pieces obey this restriction. When in check, remove the threat by a legal king move, capture, or block. Kings may not stand adjacent. The king is never captured: checkmate ends the game.
- Castling requires the relevant FEN castling right, the king and rook on their original squares, and empty squares between them. The king must not be in check or pass through or finish on an attacked square. Standard castling is encoded as the king's move: `e1g1`, `e1c1`, `e8g8`, or `e8c8`.
- En passant is available only immediately after an opposing pawn advances two squares and only when the position permits it. Move your pawn diagonally to the en passant target and remove the opposing pawn from its adjacent square. The resulting position must still leave your king safe.
- Checkmate means the active side is in check with no legal move; stalemate means it has no legal move and is not in check. Draws can also arise from repetition, move-count rules, or insufficient mating material. The harness determines the result; do not invent a move for a finished position.

## Return one move

Represent the selected move in lowercase UCI: origin square followed by destination square, with a promotion suffix `q`, `r`, `b`, or `n` when required. Examples of syntax are `e2e4`, `g1f3`, and `a7a8q`; they are not recommendations for the current position.

Follow the output protocol supplied by the harness. If it provides a strict JSON schema, return the move in its `move` field. If it provides the `choose_move` tool, call it exactly once with the selected `move`. Otherwise respond with the bare UCI move only.

Do not add explanations, SAN notation, Markdown fences, multiple alternatives, `0000`, resignation, or draw offers. Before answering, check the move against the current board, active color, and king safety.
