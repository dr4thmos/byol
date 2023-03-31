

```python
from torchvision.utils import save_image
```


```python
for idx, img in enumerate(X):
  save_image(img, 'data/debug/img'+ str(idx) + '.png')
```

```python
for idx, (x, y) in enumerate(zip(X, x_gen)):
    loss = loss_fn(x, y)
    print("idx: {} loss: {}".format(idx, loss))
```