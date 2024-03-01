import onnxruntime
print(onnxruntime.get_device())

from g2pw import G2PWConverter
conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)

print(conv("理論上只須修改此行"))