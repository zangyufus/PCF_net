f = open("Area_all_room_21_pred_gt.txt", "r+")
countwintre=0
countwinflse=0
countwalltre=0
countwallflse=0
predwin = 0
predwall = 0
predbalcony = 0
predoor = 0
gtwin = 0
gtwall = 0
gtbalcony = 0
gtdoor = 0
for line in f.readlines():
    line = line.strip().split()
    # if line[7] == "0" and line[6] == "0":
    #     countwintre = countwintre + 1
    # if line[7] == "1" and line[6] == "1":
    #     countwalltre = countwalltre + 1
    # if line[7] == "0" and line[6] == "1":
    #     countwinflse = countwinflse +1
    # if line[7] == "1" and line[6] == "0":
    #     countwallflse = countwallflse + 1
    if line[6] == '0':
        predwin = predwin + 1
    if line[6] == '1':
        predwall = predwall + 1
    if line[6] == '2':
        predbalcony = predbalcony + 1
    if line[6] == '3':
        predoor = predoor + 1
    if line[7] == '0':
        gtwin = gtwin + 1
    if line[7] == '1':
        gtwall = gtwall + 1
    if line[7] == '2':
        gtbalcony = gtbalcony + 1
    if line[7] == '3':
        gtdoor = gtdoor + 1

print(predwin,predwall,predbalcony, predoor)
print(gtwin,gtwall,gtbalcony, gtdoor)