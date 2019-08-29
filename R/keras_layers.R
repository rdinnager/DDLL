keras_custom_model_template <- function() {
  cat('iris_regression_model <- function(name = NULL) {
    
    keras_model_custom(name = name, function(self) {
    
    # define any number of layers here
    self$dense1 <- layer_dense(units = 32)
    self$dropout <- layer_dropout(rate = 0.5)
    self$dense2 <- layer_dense(units = 1)
    
    # this is the "call" function that defines what happens when the model is called
    function (x, mask = NULL) {
      x %>% 
        self$dense1() %>%
        self$dropout() %>%
        self$dense2()
    }
  })
  }
  
model <- iris_regression_model()')
  
}

keras_custom_layer_template <- function() {
  
  cat('CustomLayer <- R6::R6Class("CustomLayer",
                                  
  inherit = KerasLayer,
  
  public = list(
    
    ## add slots for any variables used inside model, e.g. self$x 
    output_dim = NULL,
    
    kernel = NULL,
    
    ## initialize any input independent variables within layer
    initialize = function(output_dim) {
      self$output_dim <- output_dim
    },
    
    ## initialize any input dependent variables within layer
    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = "kernel", 
        shape = list(input_shape[[2]], self$output_dim),
        initializer = initializer_random_normal(),
        trainable = TRUE
      )
    },
    
    ## write the logic of the layer here (e.g. the "forward pass")
    call = function(x, mask = NULL) {
      k_dot(x, self$kernel)
    },
    
    ## if output shape is different from input, output it here
    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
  )
)

## instantiate layer using create_layer, can pass base layer parameters here such as trainable, or training  
layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
  create_layer(CustomLayer, object, list(
    output_dim = as.integer(output_dim),
    name = name,
    trainable = trainable
  ))
}')
  
}


SubPixelCov2d <- R6::R6Class("SubPixelCon2d",
                           
                           inherit = KerasLayer,
                           
                           public = list(
                             
                             output_dim = NULL,
                             
                             scale = NULL,
                             
                             initialize = function(scale) {
                               self$scale <- scale
                             },
                             
                             build = function(input_shape) {
                               print(input_shape)
                               dims <- c(input_shape[1],
                                       input_shape[2] * self$scale,
                                       input_shape[3] * self$scale,
                                       as.integer(input_shape[4] / (self$scale ^ 2)))
                               self$output_dim <- tuple(dims)
                               
                             },
                             
                             call = function(x, mask = NULL) {
                               tf$depth_to_space(x, self$scale)
                               
                             },
                             
                             compute_output_shape = function(input_shape) {
                               list(input_shape[[1]], self$output_dim)
                             }
                           )
)

layer_subpixel_conv2d <- function(object, scale = 2, name = NULL, trainable = TRUE) {
  create_layer(SubPixelCov2d, object, list(
    scale = as.integer(scale),
    name = name,
    trainable = trainable
  ))
}

