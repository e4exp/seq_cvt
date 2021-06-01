import torch


def gradient_penalty(args, model, visual_emb, x_hat, w=10):
    """
    gradient penaltyを計算する

    Parameters
    ----------
    model : EDModel or CodeDiscriminator
        gradient penaltyを適用したい識別器(D)またはコード識別器(CD)のインスタンス
    x : torch.Tensor
        訓練データ
    x_fake : torch.Tensor
        生成器による生成データ
    w : int
        gradient penaltyの重要度

    Returns
    -------
    penalty : float
        gradient penaltyの値
    """
    _eps = 1e-15

    # # xと同じ次元を作成
    # alpha_size = tuple((len(x), *(1, ) * (x.dim() - 1)))
    # alpha_t = torch.Tensor
    # alpha = alpha_t(*alpha_size).to(args.device).uniform_()
    # # ランダムな係数で補間する
    # x_hat = (x.data * alpha + x_fake.data * (1 - alpha)).requires_grad_()

    def eps_norm(x):
        """
        L2ノルムを計算する

        Parameters
        ----------
        x : torch.Tensor
            入力データ

        Returns
        -------
        torch.Tensor
            入力のL2ノルム
        """
        x = x.view(len(x), -1)
        return (x * x + _eps).sum(-1).sqrt()

    def bi_penalty(x):
        """
        入力と1との二乗誤差を計算する

        Parameters
        ----------
        x : torch.Tensor
            入力データ

        Returns
        -------
        torch.Tensor
            計算された二乗誤差
        """
        return (x - 1)**2

    # x_hatに関するDの勾配を計算
    grad_x_hat = torch.autograd.grad(model(visual_emb, x_hat).sum(),
                                     [visual_emb, x_hat],
                                     create_graph=True,
                                     only_inputs=True,
                                     allow_unused=True)
    #print(grad_x_hat)
    grad_x_hat = grad_x_hat[0]
    #if grad_x_hat is None:
    #    return torch.tensor([0.0]).to(args.device, non_blocking=True)

    # 勾配のnormを1にするためのペナルティ項を計算
    penalty = w * bi_penalty(eps_norm(grad_x_hat)).mean()
    return penalty