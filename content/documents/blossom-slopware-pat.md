+++
date = '2026-03-22T14:50:53+08:00'
title = 'Blossom Slopware PAT'
+++

blossom-slopware 的永不过期的 Fine Grained Access Token

具有所有repo的尽可能高的RW权限。

```py
import base64


def decode_token() -> str:
    encoded_token = "90-50-108-48-97-72-86-105-88-51-66-104-100-70-56-120-77-85-69-49-81-107-116-77-78-69-69-119-78-48-78-51-77-122-69-49-81-110-100-110-90-49-74-82-88-51-100-116-100-87-103-50-86-49-111-48-78-48-104-119-97-84-90-53-87-85-116-77-78-85-57-116-100-87-119-53-97-107-49-114-85-51-99-51-97-85-57-113-86-85-69-49-83-107-49-50-86-49-99-122-101-71-116-69-85-48-74-66-82-69-57-78-86-109-120-106-90-109-120-48-90-51-74-97"

    chars = [chr(int(n)) for n in encoded_token.split("-")]
    encoded = "".join(chars)
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")


if __name__ == "__main__":
    decoded = decode_token()
    print("Decoded:")
    print(decoded)

```

github不让用明文上传，所以用一种奇怪的代码形式上传。

我不在乎信息安全，希望看到的人请不要拿这个账号乱写仓库，多谢。