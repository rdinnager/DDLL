#' Generate Keras Custo Model Template
#'
#' @return
#' @export
#'
#' @examples
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

#' Generate Keras Custom Layer Template
#'
#' @return
#' @export
#'
#' @examples
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
                               self$output_dim <- reticulate::tuple(dims)
                               
                             },
                             
                             call = function(x, mask = NULL) {
                               tensorflow::tf$depth_to_space(x, self$scale)
                               
                             },
                             
                             compute_output_shape = function(input_shape) {
                               list(input_shape[[1]], self$output_dim)
                             }
                           )
)


#' Keras layer to do a subpixel CNN, upsampling layer
#'
#' This layer does upsampling using a subpixel CNN, sometimes called a pixel shuffle. Essentially it shuffles
#' together channels such that each new pixel created uses one of the channels in for that pixel in the previous
#' layer.
#'
#' @param object A tensor input to the layer
#' @param scale Upsample by this much (2 = 2x)
#' @param name Optional name for the layer 
#' @param trainable Is the layer trainable?
#'
#' @return A tensor output
#' @export
layer_subpixel_conv2d <- function(object, scale = 2, name = NULL, trainable = TRUE) {
  keras::create_layer(SubPixelCov2d, object, list(
    scale = as.integer(scale),
    name = name,
    trainable = trainable
  ))
}


InstanceNormalizationParamsAsInput <- R6::R6Class("InstanceNormalizationParamsAsInput",
                           
                           inherit = KerasLayer,
                           
                           public = list(
                             
                             ## add slots for any variables used inside model, e.g. self$x 
                             output_params = NULL,
                             
                             epsilon = NULL,
                             
                             ## initialize any input independent variables within layer
                             initialize = function(output_params, epsilon) {
                               self$output_params <- output_params
                               self$epsilon <- epsilon
                             },
                             
                             ## initialize any input dependent variables within layer
                             # build = function(input_shape) {
                             #   
                             # },
                             
                             ## write the logic of the layer here (e.g. the "forward pass")
                             call = function(x, mask = NULL) {
                               
                               #c(images, beta, gamma) %<-% x
                               m <- k_mean(x$images, c(2, 3), keepdims = TRUE)
                               s <- k_std(x$images, c(2, 3), keepdims = TRUE) + self$epsilon
                               
                               normed <- (x$images - m) / s  
                               normed <- (x$gamma * normed) + x$beta
                               
                               if(self$output_params) {
                                 return(list(images = normed, beta = beta, gamma = gamma))
                               } else {
                                 return(normed)
                               }
                             }#,
                             
                             # ## if output shape is different from input, output it here
                             # compute_output_shape = function(input_shape) {
                             #   list(input_shape[[1]], self$output_dim)
                             # }
                           )
)


#' Instance Normalization Layer where the Parameters are passed in as inputs
#' 
#' This layer performs an instance normalization using parameters passed in the inputs. This is
#' useful if you want to calculate the affine transformation parameters somewhere else in the network,
#' so this should keep the backpropagation working fin.
#'
#' @param object Named list of input tensors. Should be named "images", "beta", and "gamma", for the image
#' tensor, and the beta and gamma affine parameters respectively.
#' @param output_params Should the affine parameters be outputted as well? Or just the normalized image? Default: TRUE.
#' @param epsilon epsilon value. Small value to add for numerical stability.
#' @param name Optional layer name
#' @param trainable Is the layer trainable?
#'
#' @return A list of output tensors, or a single tensor.
#' @export
#'
#' @examples
layer_instance_norm_params_as_input <- function(object, output_params = TRUE, epsilon = 1e-3, name = NULL, trainable = TRUE) {
  create_layer(InstanceNormalizationParamsAsInput, object, list(
    output_params = output_params,
    epsilon = epsilon,
    name = name,
    trainable = trainable
  ))
}