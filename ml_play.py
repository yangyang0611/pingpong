import pickle
from os import path

import numpy as np;

from mlgame.communication import ml as comm


def ml_loop(side: str):
    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    filename1 = path.join(path.dirname(__file__), 'save', 'NORMAL_model.pickle')
    filename2 = path.join(path.dirname(__file__), 'save', 'half_model.pickle')
    filename3 = path.join(path.dirname(__file__), 'save', 'IL_model.pickle')
    filename4 = path.join(path.dirname(__file__), 'save','IR_model.pickle')

    #filename4 = path.join(path.dirname(__file__), 'save', 'QQ_model.pickle')
    with open(filename1, 'rb') as file:
        clf1 = pickle.load(file)
    with open(filename2, 'rb') as file:
        clf2 = pickle.load(file)
    with open(filename3, 'rb') as file:
        clf3 = pickle.load(file)
    with open(filename4, 'rb') as file:
        clf4 = pickle.load(file)

    global ball_spe_change
    global type



    def get_direction(speed_x, speed_y):
        if (speed_x >= 0 and speed_y >= 0):
            return 0
        elif (speed_x > 0 and speed_y < 0):
            return 1
        elif (speed_x < 0 and speed_y > 0):
            return 2
        elif (speed_x < 0 and speed_y < 0):
            return 3

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()
    ball_spe_change_x = 7
    ball_spe_change_y = -7
    type = 0
    #m=0
    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()
        feature = []
        feature.append(scene_info["ball"][0])
        feature.append(scene_info["ball"][1])
        feature.append(scene_info["platform_1P"][0] + 20)
        feature.append(scene_info["ball_speed"][0])
        feature.append(scene_info["ball_speed"][1])
        feature.append(get_direction(feature[3], feature[4]))

        feature1 = np.array(feature)
        feature1 = feature1.reshape((-1, 6))

        feature.append(scene_info["blocker"][0])
        feature.append(scene_info["blocker"][1])

        feature2 = np.array(feature)
        feature2 = feature2.reshape((-1, 8))

        if ball_spe_change_y != 0 and ball_spe_change_x != 0:
            if scene_info["ball_speed"][1] / ball_spe_change_y < 0 and scene_info["ball"][1] == 260 \
                    and scene_info["blocker"][0] + 30 >= scene_info["ball"][0] >= scene_info["blocker"][0] - 5:
                type = 1
            elif abs(scene_info["ball_speed"][0]) > scene_info["ball_speed"][1] > 0:
                type = 2
            elif scene_info["ball_speed"][0]/ball_spe_change_x <0 and scene_info["ball"][1]==80:
                type=3
            elif scene_info["ball_speed"][1] / ball_spe_change_y < 0 and scene_info["ball"][1] == 415:
                type = 0

            '''elif (scene_info["ball_speed"][0] / ball_spe_change_x < 0) and (
                                scene_info["ball"][0] == scene_info["blocker"][0] - 5
                                or scene_info["ball"][0] == scene_info["blocker"][0] + 30) and 240 < scene_info["ball"][1] < 260:
                            type = 2'''
        ball_spe_change_x = scene_info["ball_speed"][0]
        ball_spe_change_y = scene_info["ball_speed"][1]
        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False
            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            # comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_LEFT)
            type = 0
            ball_served = True
        else:

            if type == 1:
                print(1)
                y = clf2.predict(feature1)

            elif type ==2:
                y = clf3.predict(feature2)
                print(2)
            elif type == 3:
                y = clf4.predict(feature2)
                print(3)
            else:
                print(4)
                y = clf1.predict(feature2)
            '''elif type == 2:
                            y = clf4.predict(feature1)
                            print(2)'''

            if y == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
                # print('NONE')
            elif y == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                # print('LEFT')
            elif y == 2:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                # print('RIGHT')
