import os

i=2001
for fileName in os.listdir("images/training_set/advance_charge_train"):
    os.rename("images/training_set/advance_charge_train/"+fileName,"images/training_set/advance_charge_train_%s.jpg"%i)
    i=i+1

j=2
for fileName in os.listdir("images/test_set/advance_charge_test"):
    os.rename("images/test_set/advance_charge_test/"+fileName,"images/test_set/advance_charge_test_%s.jpg"%j)
    j=j+1
