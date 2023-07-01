import os


def create_txt(txt_name, videos_path, labels_path):
    txt_path = os.path.join("data_txt", txt_name + ".txt")
    with open(txt_path, "a") as file:
        videos = os.listdir(videos_path)
        for video in videos:
            num_images = len(os.listdir(os.path.join(videos_path, video)))
            label = find_label(os.path.join(labels_path, video))
            file.write(os.path.join(videos_path, video) + " " + str(num_images) + " " + str(label) + "\n")
    print(txt_name + ".txt文件创建完毕")


def find_label(labels_path):
    labels = os.listdir(labels_path)
    labels_vel = []
    for label in labels:
        label_way = os.path.join(labels_path, label)
        label = int((float(open(label_way).read().strip())))
        labels_vel.append(label)
    max_label = max(labels_vel)
    if 4 <= max_label <= 5:
        max_label = 4
    if 6 <= max_label <= 15:
        max_label = 5
    return max_label


def create_test_txt(person_id, videos_path, labels_path):
    txt_path = os.path.join("data_txt", person_id + ".txt")
    with open(txt_path, "a") as file:
        videos = os.listdir(videos_path)
        for video in videos:
            # print(video[0:5])
            if video[0:5] == person_id:
                num_images = len(os.listdir(os.path.join(videos_path, video)))
                label = find_label(os.path.join(labels_path, video))
                file.write(os.path.join(videos_path, video) + " " + str(num_images) + " " + str(label) + "\n")
    print(person_id + ".txt文件创建完毕")


if __name__ == '__main__':
    videos_path = os.path.join("Pain_Images", "Images_new")
    labels_path = os.path.join("Pain_Images", "Labels_new")
    people_id = ["ll042", "jh043", "jl047", "aa048", "bm049", "dr052", "fn059", "ak064", "mg066", "bn080", "ch092",
                 "tv095", "bg096", "gf097", "mg101", "jk103", "nm106", "hs107", "th108", "ib109", "jy115", "kz120",
                 "vw121", "jh123", "dn124"]
    create_txt("rawframe", videos_path, labels_path)
    for person_id in people_id:
        create_test_txt(person_id, videos_path, labels_path)
