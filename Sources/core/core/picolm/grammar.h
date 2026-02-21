#ifndef GRAMMAR_H
#define GRAMMAR_H

#include "tokenizer.h"

/* Grammar-constrained sampling for reliable JSON output.
 *
 * When enabled, masks logits before sampling to ensure the output
 * is always syntactically valid JSON. This is critical for PicoClaw
 * tool calling where even a tiny model needs to produce parseable
 * structured output.
 *
 * The constraint tracks:
 *   - Brace/bracket depth (must balance)
 *   - String state (inside vs outside quotes)
 *   - Escape sequences
 *   - Forces output to start with '{' and end with balanced '}'
 */

typedef enum {
    GRAMMAR_NONE = 0,
    GRAMMAR_JSON = 1,
} grammar_mode_t;

typedef struct {
    grammar_mode_t mode;

    /* JSON state tracking */
    int brace_depth;     /* { } nesting */
    int bracket_depth;   /* [ ] nesting */
    int in_string;       /* inside a quoted string */
    int escape_next;     /* next character is escaped */
    int started;         /* have we output the opening brace? */

    /* Pre-computed per-token info for fast masking */
    int8_t *token_brace_delta;    /* net { minus } per token */
    int8_t *token_bracket_delta;  /* net [ minus ] per token */
    uint8_t *token_first_byte;    /* first byte of each token's string */
    int8_t *token_has_unmatched_quote; /* odd number of unescaped quotes */

    int vocab_size;
    int eos_id;
} grammar_state_t;

/* Initialize grammar state. Call after tokenizer_load.
 * Pre-computes per-token metadata for fast masking. */
void grammar_init(grammar_state_t *g, grammar_mode_t mode,
                  const tokenizer_t *tok);

/* Apply grammar constraints to logits before sampling.
 * Sets invalid tokens to -infinity. */
void grammar_apply(grammar_state_t *g, float *logits, int vocab_size);

/* Update grammar state after a token is committed.
 * Must be called with each generated token. */
void grammar_advance(grammar_state_t *g, const tokenizer_t *tok, int token);

/* Returns 1 if grammar output is complete (balanced and non-empty). */
int grammar_is_complete(const grammar_state_t *g);

/* Free grammar resources. */
void grammar_free(grammar_state_t *g);

#endif /* GRAMMAR_H */
