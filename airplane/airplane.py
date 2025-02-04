import time
import cv2
from ultralytics import YOLO
import numpy as np
from statistics import mean

model = YOLO(r'C:\Users\edayu\PycharmProjects\Yapayzeka\airplane\best (1).pt')

video_path = r'C:\Users\edayu\PycharmProjects\Yapayzeka\airplane\vlc-record-2024-06-26-18h29m32s-VDGS 40 Park ve Hizmet.mp4-.mp4'
cap = cv2.VideoCapture(video_path)

output_path = r'C:\Users\edayu\PycharmProjects\Yapay zeka\airplane\output_video_kopru2.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

windows_lst = []
windows_lst_last = []
control_lst = []
kont_kont = []

bridge_lst = []
bridge_lst_last = []
bridge_control_lst = []
bridge_kont_kont = []
bridge_detected = False
bridge_stop_time = None
bridge_has_moved = False
bridge_fps_frame = 0

i = 0
z = 0
k = 0
a = None
fps_frame = 0
start_time = time.time()
fps_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    z = z+1
    sure_gercek = z / int(fps)
    cont_bridge = 0
    for result in results[0].boxes.data:
        x1, y1, x2, y2 = map(int, result[:4])
        conf = result[4].item()
        cls = int(result[5].item())
        sure = i / int(fps)

        if model.names[cls] == 'window':
            i += 1
            windows_lst.append([x1, y1, x2, y2])
            if i % 10 == 0 and i != 0:
                windows_lst_last.append(windows_lst)
                if i != 10:
                    kontrol_deger = np.sum(np.abs(np.mean(windows_lst, axis=0) - np.mean(windows_lst_last[-2], axis=0)))
                    if kontrol_deger < 100:
                        control_lst.append(kontrol_deger)
                    if kontrol_deger < 3 and a is None:
                        a = time.time()
                        fps_frame = sure
                    if len(control_lst) == 10:
                        if mean(control_lst) < 3:
                            if len(kont_kont) == 5:
                                if min(kont_kont) < 2:
                                    print('Uçak iniyor')
                                else:
                                    a = None
                            else:
                                kont_kont.append(mean(control_lst))
                            control_lst = []
                        else:
                            control_lst = []
                            a = None
                windows_lst = []

        if model.names[cls] == 'bridge':
            cont_bridge += 1
            if conf > 0.7 and int(fps_counter)>40:
                if cont_bridge <= 1:
                    k += 1
                    bridge_lst.append([x1, y1, x2, y2])
                    if k % 10 == 0 and k != 0:
                        bridge_lst_last.append(bridge_lst)
                        if k != 10:
                            bridge_deger = np.sum(np.abs(np.mean(bridge_lst, axis=0) - np.mean(bridge_lst_last[-2], axis=0)))
                            if bridge_deger > 6 and len(bridge_kont_kont) == 0:
                                bridge_has_moved = True
                                bridge_detected = False
                                bridge_stop_time = None
                            if bridge_has_moved and bridge_deger <= 6:
                                if not bridge_detected:
                                    bridge_detected = True
                                    bridge_stop_time = time.time()
                                    bridge_fps_frame = sure
                                elif bridge_detected and bridge_stop_time == None:
                                    bridge_stop_time = time.time()
                                    bridge_fps_frame = sure
                            if bridge_detected:
                                bridge_control_lst.append(bridge_deger)
                                if len(bridge_control_lst) == 10:
                                    if mean(bridge_control_lst) < 9:
                                        if len(bridge_kont_kont) == 5:
                                            bridge_kont_kont.pop(0)
                                            bridge_kont_kont.append(mean(bridge_control_lst))
                                            if min(bridge_kont_kont) < 3:
                                                print('Köprü durdu ve bağlandı')
                                            else:
                                                bridge_stop_time = None
                                        else:
                                            bridge_kont_kont.append(mean(bridge_control_lst))
                                        bridge_control_lst = []
                                    else:
                                        bridge_control_lst = []
                                        bridge_stop_time = None
                        bridge_lst = []

    sure = i / int(fps)
    cv2.putText(frame, f'Pistteki Toplam Sure: {sure_gercek:.2f}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if a is not None:
        counter = time.time() - a
        fps_counter = sure - fps_frame
        if counter >= 5:
            frame = cv2.putText(frame, f'Park Suresi: {fps_counter:.2f}s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame = cv2.putText(frame, f'Park Suresi: 0.00s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        frame = cv2.putText(frame, f'Park Suresi: 0.00s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if bridge_stop_time is not None:
        bridge_counter = time.time() - bridge_stop_time
        bridge_fps_counter = sure - bridge_fps_frame
        if bridge_counter >= 15:
            frame = cv2.putText(frame, f'Kopru Durus Suresi: {bridge_fps_counter:.2f}s', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame = cv2.putText(frame, f'Kopru Durus Suresi: 0.00s', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        frame = cv2.putText(frame, f'Kopru Durus Suresi: 0.00s', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pistte_toplam_sure = sure_gercek
park_toplam_sure = fps_counter if a is not None else 0
kopru_durus_toplam_sure = bridge_fps_counter if bridge_stop_time is not None else 0

print(f"Pistteki Toplam Süre: {pistte_toplam_sure:.2f} saniye")
print(f"Park Süresi: {park_toplam_sure:.2f} saniye")
print(f"Köprü Duruş Süresi: {kopru_durus_toplam_sure:.2f} saniye")

cap.release()
out.release()
cv2.destroyAllWindows()
