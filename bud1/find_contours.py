import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import os
from math import sqrt
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
from scipy import stats


def canny_edge(in_filename):
    img = cv2.imread(in_filename, 0)
    gray_filtered = cv2.bilateralFilter(img, 10, 272, 2)

    # Using the Canny filter to get contours
    edges = cv2.Canny(gray_filtered, 20, 30)
    # Using the Canny filter with different parameters
    edges_high_thresh = cv2.Canny(gray_filtered, 60, 120)
    # Stacking the images to print them together
    # For comparison
    images = np.hstack((img, edges, edges_high_thresh))
    # Output the resulting

    cv2.imwrite('images/canny_img/canny_' + \
                os.path.splitext(os.path.basename(in_filename))[0] + \
                '.png', images)
    return edges_high_thresh


def contours_metadata(contours):
    ##將contours的metadata存入contours_data
    contours_data = {}
    cnts_perimeter = []
    cnts_area = []
    cnts_index = []

    ## 計算資料分布改框的條件，因為小於200的資料太多，故刪除
    for c in contours:
        if cv2.contourArea(c) > 200.0 and \
                cv2.arcLength(c, False) < 600:
            cnts_index.append(c)
            cnts_perimeter.append(cv2.arcLength(c, False))
            cnts_area.append(cv2.contourArea(c))
    contours_data['index'] = cnts_index
    contours_data['perimeter'] = cnts_perimeter
    contours_data['area'] = cnts_area
    # show_statistic(contours_data['area'],25.0)

    # plt.hist(contours_data['area'], color = 'blue', edgecolor = 'black'
    #          ,bins = int((max(contours_data['area'])-min(contours_data['area']))/25.0))
    contours_data['area_avg'] = np.average(contours_data['area'])
    contours_data['area_variance'] = np.var(contours_data['area'])
    contours_data['area_std'] = np.std(contours_data['area'])
    bp_dict = plt.boxplot(contours_data['area'])

    contours_data['area_Q1'] = [item.get_ydata()[1] for item in bp_dict['boxes']][0]
    contours_data['area_Q3'] = [item.get_ydata()[3] for item in bp_dict['boxes']][0]
    contours_data['area_median'] = [item.get_ydata()[1] for item in bp_dict['medians']][0]
    contours_data['area_min'] = [item.get_ydata()[1] for item in bp_dict['whiskers']][0]
    contours_data['area_max'] = [item.get_ydata()[1] for item in bp_dict['whiskers']][1]

    plt.clf()
    plt.close()
    # print("25%: ",contours_data['area_Q1'])
    # print("75%: ",contours_data['area_Q3'])
    # print("median: ",contours_data['area_median'])
    # print("min",contours_data['area_min'])
    # print("max",contours_data['area_max'])

    return contours_data


def myfunc(x):
    return slope * x + intercept


# x, y: data to fit
# pyplot_formatter: pyplot Format Strings
# fit_type: one of "exp", "poly", "linear"
# "poly" will use 3rd degree polynomial to draw the fit curve

def draw_fitline(ax, x, y, pyplot_formatter, fit_type):
    order = 3
    dot_num = 100
    xp = np.linspace(x[0], x[-1], dot_num)

    if fit_type == "exp":
        pexp = np.poly1d(np.polyfit(x, np.log(y), 1))
        ax.plot(xp, np.exp(pexp(xp)), '--')

    elif fit_type == "linear":
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        y_regression = [slope * n + intercept for n in x]
        ax.plot(x, y_regression)

    elif fit_type == "poly":
        p3 = np.poly1d(np.polyfit(x, y, order))
        ax.plot(xp, p3(xp), pyplot_formatter)


def BudContours():
    directory = "images/edged_img/"
    datetime_objects = []
    files = []
    banana_volume_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # files.append(os.path.join(directory,filename))
            # name, file_extension = os.path.splitext(filename)
            # last_underscore_index = name.rfind('_')
            # date_info = name[0:last_underscore_index]
            # banana_seq = int(name[-1])
            datetime_objects.append(datetime.strptime(filename, "image_%d-%m-%Y_%I-%M-%S_%p.png"))
            # datetime_objects.append(datetime.strptime(date_info,"%m_%d_%H_%M_%S"))
            # print("--",os.path.join(directory,filename))

    datetime_objects = sorted(datetime_objects)
    for d in datetime_objects:
        filename = d.strftime("image_%d-%m-%Y_%I-%M-%S_%p.png")
        files.append(os.path.join(directory, filename))
        # print("--",os.path.join(directory,filename))

    for f in files:
        edges = canny_edge(f)
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        # img = cv2.imread(f,0)
        # ret,thresh = cv2.threshold(img,127,255,0)
        # blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # value, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        metadata = contours_metadata(contours)
        minimum = metadata['area_min']
        median = metadata['area_median']
        maximum = metadata['area_max']
        area_avg = metadata['area_avg']
        area_std = metadata['area_std']
        q1 = metadata['area_Q1']
        cnt_with_area = []
        total_area = 0.0
        total_volume = 0.0
        # /** 針對average、standard deviation去篩選contours */
        for c in contours:
            # if cv2.contourArea(c)>(minimum)and \
            # cv2.contourArea(c)<(maximum):
            cnt_with_area.append(c)
            a = cv2.contourArea(c)
            total_area += a
            total_volume += sqrt(a) ** 3
        # print(len(cnt_with_area))

        # /** 使用Q1、medina、Q3來篩選contours
        # for c in contours:
        #     if cv2.contourArea(c)> median and \
        #         cv2.contourArea(c)< maximum :
        #         cnt_with_area.append(c)
        #         total_area += cv2.contourArea(c)

        read_filename = directory + \
                        os.path.splitext(os.path.basename(f))[0] + \
                        '.png'
        print(read_filename)
        out_filename = 'images/result_pics/res_' + os.path.splitext(os.path.basename(f))[0] + '.png'
        result = cv2.drawContours(cv2.imread(read_filename), cnt_with_area, -1, (0, 0, 255), 2)
        cv2.imwrite(out_filename, result)

        ## 畫圖
        # if len(cnt_with_area)!=0.0:
        # print(total_area)
        # print(cnt_with_area)
        avg_area = total_area / len(cnt_with_area)

        banana_volume = sqrt(avg_area) ** 3
        avg_volume = total_volume / len(cnt_with_area)
        # banana_volume_list.append(banana_volume)
        # banana_volume_list.append(avg_volume)
        # banana_volume_list.append(total_volume)
        # banana_volume_list.append(total_area)
        banana_volume_list.append(avg_area)

    x = list(range(1, len(banana_volume_list) + 1))

    x = np.array([i.toordinal() for i in datetime_objects])
    y_dots = np.array(banana_volume_list)
    print(y_dots)
    ###filter array
    # if you don't want data be filtered
    # you can always comment these lines
    filter_array = y_dots > 0

    x = x[filter_array]
    y_dots = y_dots[filter_array]
    ###filter array end

    ax = plt.gca()
    formatter = mdates.DateFormatter("%b")
    ax.xaxis.set_major_formatter(formatter)

    locator = mdates.MonthLocator()
    ax.xaxis.set_major_locator(locator)

    formatter = mdates.DateFormatter("%d")
    ax.xaxis.set_minor_formatter(formatter)

    locator = mdates.DayLocator()
    ax.xaxis.set_minor_locator(locator)

    left_range = min(datetime_objects) - timedelta(days=1)
    right_range = max(datetime_objects) + timedelta(days=1)

    # ax.set_xlim([datetime(2020, 6, 10), datetime(2020, 7, 1)])
    ax.set_xlim([left_range, right_range])

    ax.scatter(x, y_dots)

    draw_fitline(ax, x, y_dots, '-', 'poly')

    plt.savefig("scatter.png")

    plt.show()