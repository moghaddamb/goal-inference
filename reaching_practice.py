"""This module implements functions for running a psychophysical experiments """
import os
import itertools as it
import collections as co
import random
import cv2
import numpy as np
import pandas as pd

def get_action_time_stamp(path):
    """TODO: Docstring for get_action_time_stamp.

    @param path TODO
    @return: TODO

    """
    names = os.listdir(path)
    names = [n.split(".")[0] for n in names if
             os.path.isfile(os.path.join(path, n))]
    return names


# generates list of dictionaries for Experiment callable
def design(variables={}, num_repetition=1,  shuffle=True):
    """return a blocked trial conditions"""
    # variables = self._variables.keys()
    names, values = list(zip(*[name_value for name_value
                            in variables.items()]))
    design_values = list(it.product(*values))
    design = [dict(list(zip(names, v))) for v in design_values]

    design_repetition = design * num_repetition
    if shuffle:
        random.shuffle(design_repetition)
    return design_repetition


def dict_to_string(values={}, sep="_", prefix="", postfix=""):
    signature = prefix
    for (key, value) in values.items():
        signature=sep.join((signature, str(key), str(value)))

    signature = signature+postfix
    signature = signature.lstrip(sep)
    return signature


class Experiment(object):

    """Docstring for Experiment. """

    def __init__(self, name="expt", sub_num=-999, sub_init="test", **constants):
        """TODO: to be defined1.

        @param name TODO
        @param sub_num TODO
        @param sub_init TODO
        @param **constants TODO

        """
        self._name = name
        self._sub_num = sub_num
        self._sub_init = sub_init
        self._constants = constants
        self._values = co.OrderedDict()
        self._values['name'] = self._name
        self._values['subNum'] = self._sub_num
        self._values['subInit'] = self._sub_init
        self._values.update(self._constants)

    # generates the name for the outputted Experiment data (csv file)
    def get_expt_signature(self, sep="_", prefix="expt", postfix=""):
        return dict_to_string(self._values, sep, prefix, postfix)


    def __call__(self, trial=None, design={}, writer=None):
        for (i, d) in enumerate(design):
            #here is critial: each d must be passed to the trial callable
            result = trial(d)
            to_writer_dic = self._values.copy()
            to_writer_dic.update(d)
            to_writer_dic.update(result)
            writer(**to_writer_dic)
            print((len(design) - i))
        image(trial.display.draw_image_sequence(images))
        cv2.destroyAllWindows()


def get_file_name(path="", names=[], prefix="", post_fix="", connector="_"):
    file_name = prefix + connector.join([str(n) for n in names]) + post_fix
    return os.path.join(path, file_name)


def play_video(name, fps=30, window_name="image"):
    cap = cv2.VideoCapture(name)
    assert os.path.isfile(name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
        pre_frame=frame

    cap.release()
    return pre_frame


def play_images(images, fps=30, window_name="image"):
    for image in images:
        cv2.imshow(window_name, image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    return image


def draw_objects(frame, obj_se, size, color=None):
    def get_color(obj_se):
        color_dic = dict((["R", (0, 0, 255)],
                          ["G", (0, 255, 0)],
                          ["B", (255, 0, 0)],
                          ["W", (255, 255, 255)]))
        name = obj_se.Object
        return color_dic[name[0]]

    if color is None:
        color = get_color(obj_se)

    cv2.circle(frame,(int(obj_se.color_x), int(obj_se.color_y)), size, color,-1)
    return frame

def draw_text(frame, text, x=500, y=300, size=0.41, color=(255,255,255), thickness=1, line = cv2.LINE_AA):
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, line)
    return frame

def draw_rest_object(frame, color=(0, 0, 0), pos=(300, 300)):
    cv2.circle(frame, pos, 10, color, -1)


def get_nearest_to_mouse(mouse_pos_arr, obj_df):
    distance = np.linalg.norm(obj_df[["color_x", "color_y"]].values - mouse_pos_arr, axis=1)
    min_dist = np.min(distance)
    if min_dist>30:
        return None

    obj_index = np.argmin(distance)
    obj_se = obj_df.iloc[obj_index] # type -> 'pandas.core.series.Series'
    return obj_se


def select_object(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        pass


class Writer(object):

    """Docstring for Writer. """

    def __init__(self, path, replace=False):
        """TODO: to be defined.

        @param path TODO
        @param mode TODO

        """
        self._path = path
        self._replace = replace
        self._index = 0


    def __call__(self, **kwargs):
        if self._replace and self._couter ==0:
            mode = "w"
        else:
            mode = "a"

        if self._index==0:
            header = True
        else:
            header = False

        result_dict={}
        result_dict.update(kwargs)
        result_df = pd.DataFrame(data=result_dict,
                                 index=[self._index])
        result_df.to_csv(self._path, header=header, mode=mode)

        self._index+=1


class Display(object):

    """Docstring for Display. """

    def __init__(self,
                 name="window",
                 obj_path="",
                 fps=30,
                 ratio=0.5,
                 uh=0,
                 uv=0,
                 lh=0,
                 lv=0,
                 wx=0,
                 wy=0
                 ):
        """TODO: to be defined1.

        @param name TODO
        @param ratio TODO
        @param uh TODO
        @param uv TODO
        @param lh TODO
        @param lv TODO
        @param fps TODO

        """
        self._name = name
        self._uh = uh
        self._uv = uv
        self._lh = lh
        self._lv = lv
        self.fps = fps
        self.mouse_x = -999
        self.mouse_y = -999
        # self.mouse_click = False
        self.left_click = False
        self.right_click = False
        self.stop = False
        self.selection = False
        self.next = False

        self.obj_df = pd.read_excel(obj_path)
        self.obj_df.color_x*=ratio
        self.obj_df.color_y*=ratio

        self.inferred_target = None

        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._name, cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self._name, wx, wy)
        cv2.setMouseCallback(self._name, self.update_mouse)

    def __del__(self):
        cv2.destroyAllWindows()

    def crop_image(self, image):
        return image[self._uv:self._lv, self._uh:self._lh]

    def check(self):
        return cv2.waitKey(1000//self.fps) & 0xFF == ord('q')

    def update_mouse(self, event, x, y, flag, param):
        self.mouse_x = x + self._uh
        self.mouse_y = y + self._uv

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_click = True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.right_click = True


    def draw_preview(self, image):
        # print("DRAW REST OBJECT")
        # print(self.left_click)
        rest_pos = (605, 510)
        draw_rest_object(image, (0, 0, 0), (605, 510))
        cropped = self.crop_image(image)
        self.left_click = False
        while not self.left_click:
            cv2.imshow(self._name, cropped)
            self.check()
            mouse_arr = np.array([self.mouse_x, self.mouse_y])
            diff = np.array(rest_pos) - mouse_arr
            if np.linalg.norm(diff)<10 and self.left_click:
                # print("LEFT CLICK FALSE -> TRUE")
                break
            self.left_click = False
        # self.left_click = False

    def draw_image_sequence(self, images):
        # print(type(self.images))
        self.stop = False
        for image in images:
            draw_rest_object(image, (150, 150, 150), (605, 510))
            cv2.imshow(self._name, self.crop_image(image))
            if self.check():
                break
            if self.left_click == True:
                none_test = self.draw_selection(image)
            #print(none_test)
                if none_test is None:
                    continue
                else:
                    self.inferred_target = none_test
                    draw_text(image,"Thank you for your response!")
                    draw_text(image,"Now let's see the remaining video of what the person actually did.",355, 310, 0.41)
                    cv2.imshow(self._name, self.crop_image(image))
                    cv2.waitKey(4000)
                    # break
            self.left_click = False
        return image


    def draw_selection(self, frame):
        self.selection = False
        self.left_click = False
        while not self.left_click:
            curr_frame = np.copy(frame)
            if not (self.mouse_x == -1999 and self.mouse_y==-1999):
                mouse_arr = np.array([self.mouse_x, self.mouse_y], dtype=int)

                obj_se = get_nearest_to_mouse(mouse_arr, self.obj_df)

                if obj_se is not None:
                    draw_objects(curr_frame, obj_se, 9)
                else:
                    # print("STOPS HERE")
                    return None
                #print("STOPS HERE")
            cv2.imshow(self._name, self.crop_image(curr_frame))
            if self.check():
                break

        if obj_se is None:
            return None
        for i in range(1):
            draw_objects(frame, obj_se, 15)
            cv2.imshow(self._name, self.crop_image(frame))
            # if self.check():
            #     break
        self.selection = True
        self.right_click = False
        return obj_se.Object


    def draw_feedback(self, frame, target):
        obj_se = self.obj_df[self.obj_df.Object==target].iloc[0]
        for i in range (30):
            draw_objects(frame, obj_se, 15, (0, 0, 0))
            cv2.imshow(self._name, self.crop_image(frame))
            if self.check():
                break


class Trial():
    def __init__(self,
                 name="grasping",
                 image_path="",
                 skeleton_path="",
                 # event_segmentation_path="",
                 truth_df=None,
                 display = None,
                 practice=False
                 ):
        self.name = name
        self.image_path = image_path
        self.skeleton_path = skeleton_path
        self.truth_df = truth_df
        # self.event_segmentation_path = event_segmentation_path
        self.display = display
        self.inferred_target = None
        self.result = {}
        self.practice=practice



    def get_images(self, actor, event, percent):
        trial_image_path = os.path.join(self.image_path, actor, "Clip_"+str(event))
        assert os.path.isdir(trial_image_path)

        all_images = [i for i in os.listdir(trial_image_path)
                      if i.endswith(".png")]
        all_images.sort(key=lambda x:float(x.split(".")[0]))
        num_images = len(all_images)
        num_percent_images = int(num_images*percent)
        image_name_lst = all_images[0:num_percent_images]
        image_path_lst = [os.path.join(trial_image_path, i)
                          for i in image_name_lst]
        images = [cv2.imread(i) for i in image_path_lst]
        return images


    def get_single_image(self, trial_image_path, time_stamp):
        image_name = os.path.join(trial_image_path, str(time_stamp)  +".bmp")
        return cv2.imread(image_name)


    def __call__(self, design_value):
        actor = design_value["actor"]
        event = design_value["event"]
        percent = design_value["percent"]

        # get all images list in Clip_event images folder
        images = self.get_images(actor, event, percent)
        index = (self.truth_df.Actor==actor)&(self.truth_df.Event==event)
        # truth = self.truth_df[(self.truth_df.Actor==actor)&(self.truth_df.Event==event)].iloc[0].Truth
        self.result["target"] = str(self.truth_df[index].iloc[0].Truth)
        self.result["total_images"] = len(images)
        self.display.draw_preview(images[0])
        # print("START")
        last_frame = self.display.draw_image_sequence(images)

        counter = 1
        response_frame = 0
        for image in images:
            if np.array_equal(image, last_frame):
                response_frame = counter
                break
            else:
                counter += 1
        self.result["response_frame_num"] = response_frame

        inferred_target = self.display.inferred_target
        while True:
            if inferred_target is not None:
                if self.practice:
                    self.display.draw_feedback(last_frame, self.result["target"])

                self.result["inferred_target"] = inferred_target
                self.mouse_click = False
                return self.result
            else:
                new_images = images[counter:]

                for image in new_images:
                    cv2.imshow(self._name, self.crop_image(image))

                last_frame = self.display.draw_image_sequence(new_images)
                counter = counter
                response_frame = counter
                for image in new_images:
                    if np.array_equal(image, last_frame):
                        response_frame = counter
                        break
                    else:
                        counter += 1
                self.result["response_frame_num"] = response_frame
                inferred_target = self.display.inferred_target



def main():
    expt = Experiment(name="reaching_stop_any_time", sub_num="2", sub_init="test", action="reaching") # change sub_num
    project_path = "/Users/moghaddam/Desktop/Grasping-master/"
    data_path = os.path.join(project_path, "data")
    image_path = os.path.join(data_path, "images")
    skeleton_path = os.path.join(data_path, "skeletons")
    env_path = os.path.join(data_path, "environments")
    obj_path = os.path.join(env_path, "target_pos_no_obstacle.xlsx")

    event_path = os.path.join(data_path, "events")
    # event_segmentation_path = os.path.join(event_path, "segmentations")
    truth_path = os.path.join(event_path, "reaching_ground_truth.xlsx")

    # Tao   14  B4
    # Tao 15  W1
    # Shira   0   R2

    truth_df = pd.read_excel(truth_path,"Sheet1")

    actor_names = ["Danny"]
    # actor_names = ["Junli"]

    variables = dict([("actor", actor_names),
                      ("event", list(range(10))),
                      # ("percent", [0.2, 0.35, 0.5, 0.65])])
                      ("percent", [1.0])])
    design_values = design(variables=variables, num_repetition=1)

    name = expt.get_expt_signature(sep="_", postfix=".csv")
    path = os.path.join("data", name)

    writer = Writer(path=path, replace=False)

    display = Display("grasping", obj_path=obj_path, fps=9,
                      uh=354, uv=158, lh=800, lv=530)
 # uh=354, uv=158, lh=800, lv=530)
    trial=Trial(image_path=image_path,
                skeleton_path=skeleton_path,
                # event_segmentation_path=event_segmentation_path,
                truth_df = truth_df,
                display=display,
                practice=False)

    expt(trial, design_values, writer)

if __name__ == "__main__":
    main()




