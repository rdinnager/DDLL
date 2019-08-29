#' Convert a Keras layer quickly to a custom Keras model so it can be used in eager mode.
#'
#' Mainly used for testing new custom Keras layers quickly and easily by converting them to
#' custom Keras models, which can be used in tensorflow eager mode
#'
#' Don't forget to set \code{use_implementation("tensorflow")}!
#' 
#' @param layer Layer to convert to a custom model. Should be an expression, where the layer is
#' invoked with desired parameter values.
#' @param build_model Should the built custom model be returned? Or a function that can be used to build
#' the model? Default FALSE.
#'
#' @return Either a layer or a function to build the layer. 
#'
#' @export
keras_layer_to_model <- function(layer, build_model = FALSE) {
  
  layer_maker <- function(name = NULL) {
    keras::keras_model_custom(name = name, function(self) {
    
    # define any number of layers here
    self$layer <- layer
    
    # this is the "call" function that defines what happens when the model is called
    function (x, mask = NULL) {
      x %>% 
        self$layer()
    }
  })
  }
  
  if(build_model) {
    return(layer_maker())
  } else {
    return(layer_maker)
  }
  
}