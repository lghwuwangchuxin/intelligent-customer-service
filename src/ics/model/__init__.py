"""L2 model：模型访问的**唯一收敛点**。

上层永远通过别名（chat-main / chat-cheap / embed-main）访问模型，
永不写厂商模型名。换厂商时爆炸半径限在 registry.py 一个文件。
"""
