#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct {
    float    temperature;
    float    top_p;
    uint64_t rng_state;   /* xorshift64 state */
} sampler_t;

/* Initialize sampler with given parameters */
void sampler_init(sampler_t *s, float temperature, float top_p, uint64_t seed);

/* Sample a token index from logits[vocab_size].
 * Modifies logits in-place (temperature scaling, softmax). */
int sampler_sample(sampler_t *s, float *logits, int vocab_size);

#endif /* SAMPLER_H */
