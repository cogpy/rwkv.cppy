#ifndef ESN_H
#define ESN_H

#include "rwkv.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(ESN_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(ESN_BUILD)
#            define ESN_API __declspec(dllexport)
#        else
#            define ESN_API __declspec(dllimport)
#        endif
#    else
#        define ESN_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define ESN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // ESN error flags for error handling
    enum esn_error_flags {
        ESN_ERROR_NONE = 0,
        ESN_ERROR_ARGS = 1 << 8,
        ESN_ERROR_ALLOC = 1,
        ESN_ERROR_MODEL = 2,
        ESN_ERROR_TRAINING = 3,
        ESN_ERROR_PREDICTION = 4,
        ESN_ERROR_DIMENSION = 5,
        ESN_ERROR_STATE = 6
    };

    // ESN chatbot personality types
    enum esn_personality_type {
        ESN_PERSONALITY_CONSERVATIVE = 0,
        ESN_PERSONALITY_BALANCED = 1,
        ESN_PERSONALITY_CREATIVE = 2,
        ESN_PERSONALITY_CUSTOM = 3
    };

    // ESN readout layer types
    enum esn_readout_type {
        ESN_READOUT_RIDGE = 0,
        ESN_READOUT_LINEAR = 1,
        ESN_READOUT_MLP = 2,
        ESN_READOUT_ONLINE = 3
    };

    // ESN configuration structure
    struct esn_config {
        uint32_t units;                    // Number of reservoir units
        float spectral_radius;             // Reservoir spectral radius (creativity control)
        float leaking_rate;                // Memory persistence
        float input_scaling;               // Input sensitivity
        float noise_scaling;               // Response variability
        float ridge_alpha;                 // Ridge regression regularization
        uint32_t warmup_steps;             // Number of warmup steps
        enum esn_personality_type personality; // Chatbot personality
        enum esn_readout_type readout_type;    // Readout layer type
        bool online_learning;              // Enable online adaptation
        uint32_t mlp_hidden_size;          // MLP readout hidden layer size
        float learning_rate;               // Online learning rate
    };

    // ESN context for chatbot operations
    struct esn_context;

    // ESN training data structure
    struct esn_training_data {
        uint32_t** sequences;              // Input token sequences
        size_t* sequence_lengths;          // Length of each sequence
        float** targets;                   // Target outputs for each sequence
        size_t* target_lengths;            // Length of each target sequence
        size_t n_sequences;                // Number of training sequences
        size_t output_dim;                 // Output dimension
    };

    // ESN prediction results
    struct esn_prediction_result {
        float* outputs;                    // Predicted outputs
        size_t output_length;              // Length of output sequence
        size_t output_dim;                 // Output dimension
        float confidence;                  // Prediction confidence score
    };

    // ESN chatbot conversation state
    struct esn_conversation_state {
        float* reservoir_state;            // Current reservoir state
        float* conversation_history;       // Conversation context
        size_t history_length;             // History length
        uint32_t turn_count;               // Number of conversation turns
        enum esn_personality_type current_personality; // Active personality
    };

    // Initialize ESN context with RWKV model and configuration
    // Returns NULL on error
    ESN_API struct esn_context* esn_init(
        struct rwkv_context* rwkv_ctx,
        const struct esn_config* config
    );

    // Create ESN configuration with default values for personality type
    ESN_API struct esn_config esn_create_config(enum esn_personality_type personality);

    // Train the ESN readout layer with training data
    // Returns true on success, false on error
    ESN_API bool esn_train(
        struct esn_context* ctx,
        const struct esn_training_data* training_data
    );

    // Predict outputs for input token sequence
    // Returns prediction result, must be freed with esn_free_prediction
    ESN_API struct esn_prediction_result* esn_predict(
        struct esn_context* ctx,
        const uint32_t* tokens,
        size_t sequence_length
    );

    // Run reservoir without readout layer (get raw activations)
    // Returns reservoir activations, must be freed with esn_free_activations
    ESN_API float* esn_run_reservoir(
        struct esn_context* ctx,
        const uint32_t* tokens,
        size_t sequence_length,
        size_t* activation_length
    );

    // Initialize chatbot conversation
    ESN_API struct esn_conversation_state* esn_init_conversation(
        struct esn_context* ctx
    );

    // Process chatbot input and generate response
    ESN_API struct esn_prediction_result* esn_chatbot_respond(
        struct esn_context* ctx,
        struct esn_conversation_state* conv_state,
        const uint32_t* input_tokens,
        size_t input_length,
        uint32_t max_response_length
    );

    // Switch chatbot personality dynamically
    ESN_API bool esn_switch_personality(
        struct esn_context* ctx,
        struct esn_conversation_state* conv_state,
        enum esn_personality_type new_personality
    );

    // Update ESN with online learning from new interaction
    ESN_API bool esn_online_update(
        struct esn_context* ctx,
        const uint32_t* input_tokens,
        size_t input_length,
        const float* target_output,
        size_t output_length
    );

    // Get ESN internal state information
    ESN_API uint32_t esn_get_reservoir_size(const struct esn_context* ctx);
    ESN_API uint32_t esn_get_output_size(const struct esn_context* ctx);
    ESN_API enum esn_personality_type esn_get_personality(const struct esn_context* ctx);

    // Reset ESN reservoir state
    ESN_API void esn_reset_state(struct esn_context* ctx);

    // Evaluate model performance on test data
    ESN_API float esn_evaluate(
        struct esn_context* ctx,
        const struct esn_training_data* test_data
    );

    // Error handling
    ESN_API void esn_set_print_errors(struct esn_context* ctx, bool print_errors);
    ESN_API enum esn_error_flags esn_get_last_error(struct esn_context* ctx);

    // Memory management
    ESN_API void esn_free_prediction(struct esn_prediction_result* result);
    ESN_API void esn_free_activations(float* activations);
    ESN_API void esn_free_conversation_state(struct esn_conversation_state* conv_state);
    ESN_API void esn_free_training_data(struct esn_training_data* data);
    ESN_API void esn_free_context(struct esn_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // ESN_H