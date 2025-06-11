# Model's outputs/accuracies
The reference outputs are in [instructions.md](https://github.com/Yousefbahr/MinLlama/blob/master/instructions.md)

## Text Continuation
Prompt:
``` I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is ```

Generation with temperature 0:
>I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is still had auntie got stuck together.
Little figure of course, butter creative in the restoring the restoring the restoring round and said, butter creative in the restoring round and said, butter creative in the restoring round and said, butter creative in the restoring round and said, butter creative in

Generation with temperature 1:
> I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, ished up ahead and to pickingсонives keep the restoring feeling unsimssuit Remainni breakaging packageed gateidesworms can goats and Jane asked him milk départmunлееumannails for dinner Doomed great will always jolly false and hurried while kneeites are СавезнеDen handle ringing spying gate Lettersposition direction wearing cxwür

## Zero Shot Prompting 

### Zero-Shot Prompting for SST:

- Dev Accuracy: 0.182

- Test Accuracy: 0.178

### Zero-Shot Prompting for CFIMDB:

- Dev Accuracy: 0.514
  
- Test Accuracy: 0.508


## Classification Finetuning

### Finetuning for SST

- Dev Accuracy: 0.333
  
- Test Accuracy: 0.320

### Finetuning for CFIMDB

- Dev Accuracy: 0.882
  
- Test Accuracy: 0.490
