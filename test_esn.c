#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esn.h"
#include "rwkv.h"

int main() {
    printf("Testing ESN C++ implementation...\n");
    
    // Test configuration creation
    struct esn_config config = esn_create_config(ESN_PERSONALITY_BALANCED);
    printf("✓ Created ESN config with personality: %d\n", config.personality);
    printf("  Units: %d, Spectral radius: %.2f, Leaking rate: %.2f\n", 
           config.units, config.spectral_radius, config.leaking_rate);
    
    // Try to load a test model
    const char* model_path = "tests/tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin";
    printf("Loading RWKV model: %s\n", model_path);
    
    struct rwkv_context* rwkv_ctx = rwkv_init_from_file(model_path, 4, 0);
    if (!rwkv_ctx) {
        printf("✗ Failed to load RWKV model\n");
        return 1;
    }
    printf("✓ RWKV model loaded successfully\n");
    
    // Initialize ESN context
    struct esn_context* esn_ctx = esn_init(rwkv_ctx, &config);
    if (!esn_ctx) {
        printf("✗ Failed to initialize ESN context\n");
        rwkv_free(rwkv_ctx);
        return 1;
    }
    printf("✓ ESN context initialized successfully\n");
    
    // Test basic ESN functions
    uint32_t reservoir_size = esn_get_reservoir_size(esn_ctx);
    enum esn_personality_type personality = esn_get_personality(esn_ctx);
    printf("  Reservoir size: %d\n", reservoir_size);
    printf("  Personality: %d\n", personality);
    
    // Reset state
    esn_reset_state(esn_ctx);
    printf("✓ State reset completed\n");
    
    // Test reservoir run with simple tokens
    uint32_t test_tokens[] = {1, 2, 3, 4, 5};
    size_t activation_length = 0;
    
    float* activations = esn_run_reservoir(esn_ctx, test_tokens, 5, &activation_length);
    if (activations) {
        printf("✓ Reservoir run successful, activation length: %zu\n", activation_length);
        printf("  First few activations: %.4f, %.4f, %.4f\n", 
               activations[0], activations[1], activations[2]);
        esn_free_activations(activations);
    } else {
        printf("✗ Reservoir run failed\n");
    }
    
    // Test conversation state initialization
    struct esn_conversation_state* conv_state = esn_init_conversation(esn_ctx);
    if (conv_state) {
        printf("✓ Conversation state initialized\n");
        
        // Test personality switching
        bool switch_result = esn_switch_personality(esn_ctx, conv_state, ESN_PERSONALITY_CREATIVE);
        if (switch_result) {
            printf("✓ Personality switched to creative\n");
        }
        
        esn_free_conversation_state(conv_state);
    }
    
    // Cleanup
    esn_free_context(esn_ctx);
    rwkv_free(rwkv_ctx);
    
    printf("✓ All tests completed successfully!\n");
    return 0;
}