#
#  Utility Functions
#
#
import torch

def output_label(label):
    """

    A function to convert the item codes produced by the model back into text descriptions

    Args:
        label (torch.Tensor or Int): The numerical label to be mapped back into a clothing item

    Returns:
        text_mapping (str): The output mapping

    """
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]