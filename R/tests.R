library(tensorflow)

tfe_enable_eager_execution(device_policy = "silent")
tfe <- tf$contrib$eager

library(keras)

use_implementation("tensorflow")

mnist <- dataset_mnist()

image_tens <- mnist$train$x[1:10, , ] %>%
  k_expand_dims() %>%
  k_cast(dtype = "float32")

conv_2d <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    # define any number of layers here
    self$conv_2d <- layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same",
                                  use_bias = FALSE, strides = c(1, 1))
    
    # this is the "call" function that defines what happens when the model is called
    function (x, mask = NULL) {
      x %>% 
        self$conv_2d()
    }
  })
}

sub_conv_2d <- function(name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    # define any number of layers here
    self$sub_conv_2d <- layer_subpixel_conv2d(scale = 2)
    
    # this is the "call" function that defines what happens when the model is called
    function (x, mask = NULL) {
      x %>% 
        self$sub_conv_2d()
    }
  })
}

sub_conv_2d <- keras_layer_to_model(
  layer_subpixel_conv2d(scale = 2),
  build_layer = TRUE
)

model_conv2d <- conv_2d()

model_subconv2d <- sub_conv_2d()

image_tens %>%
  model_conv2d() %>%
  model_subconv2d()

image_tens %>%
  model_conv2d() %>%
  sub_conv_2d()