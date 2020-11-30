import torch
import torch.backends.cudnn

gpu = torch.device('cuda')

# =======================================================================================================================
# (1) https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
# (2) https://www.cs.toronto.edu/~graves/icml_2006.pdf
# (3) https://distill.pub/2017/ctc/
# =======================================================================================================================


# this directly follows Graves et al's paper, in contrast to the production implementation, it does not use log-space
def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # ==========================================================================================================
        # To allow for blanks in the output paths, we consider a modified label sequence l', with blanks added to the
        # beginning and the end and inserted between every pair of labels. The length of l' is therefore 2|l| + 1.
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # ==========================================================================================================
        # переход из log space
        probs = log_probs[:input_length, i].exp()
        # ==========================================================================================================
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        # инициализируем начало sequence'а вероятностью blank'а как в paper в формуле 6
        alpha[0] = probs[0, blank]
        # второй элемент инициализируем опять же как в формуле 6
        alpha[1] = probs[0, targets_prime[1]]
        # эта маска валидна только когда целевой таргет либо blank либо равен таргету на 2 шага назад
        # почему - смотри case 1 & 2 в источнике 3
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        # проходим по всей длине последовательности
        for t in range(1, input_length):
            # тут мы сначала клонируем текущую альфу
            alpha_next = alpha.clone()
            # потом сдвигаем альфу на один элемент вправо и складываем фактически каждый At + (At-1)
            alpha_next[1:] += alpha[:-1]
            # и только для тех элементов, где маска проходит, мы прибавляем еще и At-2
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            # так как мы идем слева направо, и альфа инициализирована нулями, то это тоже фактически маска
            # и теперь мы перемножаем вероятности на каждом timestep с этой маской
            # к последнему timestep последовательности здесь будет заполнен alpha[-1] как произведение вероятности на
            # сумму всей истории которая была левее него (смотри формулы 6 и 7 в источнике 2)
            # фактически это расчет стоимости пути в графе с помощью динамического программирования (смотри источник 1)
            alpha = probs[t, targets_prime] * alpha_next
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output


ctc_loss = torch.nn.CTCLoss()
target_lengths = [30, 25, 20]
input_lengths = [50, 50, 50]
targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
log_probs = torch.randn(50, 3, 15, dtype=torch.float, device=gpu).log_softmax(2)
res = ctc_loss(log_probs, targets, input_lengths, target_lengths)
expected = ctcloss_reference(log_probs, targets.cuda(), input_lengths, target_lengths).float()
with torch.backends.cudnn.flags(enabled=False):
    res2 = ctc_loss(log_probs, targets.cuda().long(), input_lengths, target_lengths)
# tensor(True, device='cuda:0')
# tensor(True, device='cuda:0')
print(torch.all(res.eq(expected)))
print(torch.all(res2.eq(res)))
