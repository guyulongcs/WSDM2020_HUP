import numpy as np


class MLTool():
    @classmethod
    def get_emb_mochang(cls, emb):
        mochang = []
        for l in emb:
            mochang.append(np.linalg.norm(l))
        return mochang