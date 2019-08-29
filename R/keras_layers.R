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

