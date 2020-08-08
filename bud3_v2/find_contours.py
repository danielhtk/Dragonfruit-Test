import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
from math import sqrt
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
from scipy import stats

def canny_edge(in_filename):
    img = cv2.imread(in_filename, 0)
    gray_filtered = cv2.bilateralFilter(img, 10, 600, 2)

    # Using the Canny filter to get contours
    edges = cv2.Canny(gray_filtered, 20, 30)
    # Using the Canny filter with different parameters
    edges_high_thresh = cv2.Canny(gray_filtered, 60 , 300)
    # Stacking the images to print them together
    # For comparison
    images = np.hstack((img, edges, edges_high_thresh))
    # Output the resulting

    cv2.imwrite('images/canny_img/canny_'+ \
                 os.path.splitext(os.path.basename(in_filename))[0] + \
                 '.png',images)
    return edges_high_thresh

def rect():
    path = "images/rect_pic"
    Path(path).mkdir(parents=True, exist_ok=True)
    res_dir = "images/result_pics/"
    images = []
    fn = []
    for filename in os.listdir(res_dir):
        img = cv2.imread(os.path.join(res_dir, filename))
        if img is not None:
            images.append(img)
            fn.append(filename)
    for i in range(len(images)) :
        # temp = images[i].astype(np.uint8)
        temp = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY);
        ret, thresh = cv2.threshold(temp, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 255, 0), 2)
            ## get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            ## convert all coordinates floating point values to in
            box = np.int0(box)
            # rect_pic = cv2.drawContours(temp, [box], 0, (0, 0, 255))
            temp = cv2.drawContours(temp, [box], -1, (0, 255, 0), 2)
        # cv2.imwrite('images/rect_pic/rect_' + \
        #             os.path.splitext(os.path.basename( np.array(fn)[i]) )[0] + \
        #             '.png', rect_pic)
        rect = cv2.drawContours(temp, contours, -1, (0, 255, 0), 2)
        cv2.imwrite('images/rect_pic/rect_' + \
                    os.path.splitext(os.path.basename( np.array(fn)[i]) )[0] + \
                    '.png', rect)

# def contours_metadata(contours):
#     ##將contours的metadata存入contours_data
#     contours_data = {}
#     cnts_perimeter = []
#     cnts_area = []
#     cnts_index = []
#
#     ## 計算資料分布改框的條件，因為小於200的資料太多，故刪除
#     for c in contours:
#         if cv2.contourArea(c) > 200.0 and \
#                 cv2.arcLength(c, False) < 600:
#             cnts_index.append(c)
#             cnts_perimeter.append(cv2.arcLength(c, False))
#             cnts_area.append(cv2.contourArea(c))
#     contours_data['index'] = cnts_index
#     contours_data['perimeter'] = cnts_perimeter
#     contours_data['area'] = cnts_area
#     # show_statistic(contours_data['area'],25.0)
#
#     # plt.hist(contours_data['area'], color = 'blue', edgecolor = 'black'
#     #          ,bins = int((max(contours_data['area'])-min(contours_data['area']))/25.0))
#     contours_data['area_avg'] = np.average(contours_data['area'])
#     contours_data['area_variance'] = np.var(contours_data['area'])
#     contours_data['area_std'] = np.std(contours_data['area'])
#     bp_dict = plt.boxplot(contours_data['area'])
#
#     contours_data['area_Q1'] = [item.get_ydata()[1] for item in bp_dict['boxes']][0]
#     contours_data['area_Q3'] = [item.get_ydata()[3] for item in bp_dict['boxes']][0]
#     contours_data['area_median'] = [item.get_ydata()[1] for item in bp_dict['medians']][0]
#     contours_data['area_min'] = [item.get_ydata()[1] for item in bp_dict['whiskers']][0]
#     contours_data['area_max'] = [item.get_ydata()[1] for item in bp_dict['whiskers']][1]
#
#     plt.clf()
#     plt.close()
#     # print("25%: ",contours_data['area_Q1'])
#     # print("75%: ",contours_data['area_Q3'])
#     # print("median: ",contours_data['area_median'])
#     # print("min",contours_data['area_min'])
#     # print("max",contours_data['area_max'])
#
#     return contours_data

def myfunc(x):
    return slope * x + intercept

# draw_fitline v0.1.1 updated at 2020/08/06
# x, y: data to fit
# pyplot_formatter: pyplot Format Strings
# fit_type: one of "exp", "poly", "linear"
# "poly" will use 3rd degree polynomial to draw the fit curve

# The original x value bases on timestamps
# and the value is too big, regression function will have ugly coefficients to display
# We use twiny() to draw on another x-axis
# to avoid the problem here
def draw_fitline(ax, x, y, pyplot_formatter, fit_type):
    def R2(x, y, coefs):
        p = np.poly1d(coefs)
        yhat = p(x)  # or [p(z) for z in x]
        ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        return ssreg / sstot

    order = 3
    dot_num = 100
    x_base = x[0]
    x = [item - x_base for item in x]
    xp = np.linspace(x[0], x[-1], dot_num)

    main_ax = ax.twiny()
    original_xlim = ax.get_xlim()
    main_ax.set_xlim([timestamp - x_base for timestamp in original_xlim])

    if fit_type == "exp":

        # Using polyfit to compute exponential regression function
        # y = b*exp(a*x)
        exp_coef = np.polyfit(x, np.log(y), 1)
        pexp = np.poly1d(exp_coef)

        # But be careful, to transform a linear function to a exponential function
        # we should addjust coefficients.
        print(exp_coef)
        a = exp_coef[0]
        b = np.exp(exp_coef[1])

        print("R2 value: ")
        print(R2(x, np.log(y), exp_coef))
        print("for function y = b*exp(a*x), b = {b:e}, a={a:e}".format(b=b, a=a))
        line_expfit, = main_ax.plot(xp, np.exp(pexp(xp)),
                                    label=r'exponential regression $y = {b:.2f}e^{{{a:.2f}x}}$'.format(b=b, a=a))
        # main_ax.legend(handles=[line_expfit])

        return line_expfit


    elif fit_type == "linear":
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        y_regression = [slope * n + intercept for n in x]

        print("R2 value: ")
        print(R2(x, y, [slope, intercept]))
        print("for function y = a*x + b, a = {a}, b = {b}".format(a=slope, b=intercept))

        line_regression, = main_ax.plot(x, y_regression,
                                        label=r'linear regression $y = {a:.2f}x + {b:.2f}$'.format(a=slope,
                                                                                                   b=intercept))
        main_ax.legend(handles=[line_regression])

        return line_regression

    elif fit_type == "poly":
        poly_coef = np.polyfit(x, y, order)
        p3 = np.poly1d(poly_coef)

        print("R2 value: ")
        print(R2(x, y, poly_coef))
        print("for function y = ax^3 + bx^2 + cx + d, \n a = {a}, \n b = {b}, \n c = {c}, \n d = {d}".format(
            a=poly_coef[3],
            b=poly_coef[2],
            c=poly_coef[1],
            d=poly_coef[0]))

        line_regression, = main_ax.plot(xp, p3(xp),
                                        label=r'polynomial regression $y = {a:.2f}x^3 + {b:.2f}x^2 + {c:.2f}x + {d:.2f}$'.format(
                                            a=poly_coef[3],
                                            b=poly_coef[2],
                                            c=poly_coef[1],
                                            d=poly_coef[0]))
        main_ax.legend(handles=[line_regression])

        return line_regression

def BudContours():
    directory = "images/edged_img/"
    datetime_objects = []
    files = []
    buds_area_list = []
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
        # metadata = contours_metadata(contours)
        # minimum = metadata['area_min']
        # median = metadata['area_median']
        # maximum = metadata['area_max']
        # area_avg = metadata['area_avg']
        # area_std = metadata['area_std']
        # q1 = metadata['area_Q1']
        cnt_with_area = []
        total_area = 0.0
        # total_volume = 0.0
        # /** 針對average、standard deviation去篩選contours */
        for c in contours:
            # if cv2.contourArea(c)>(minimum)and \
            # cv2.contourArea(c)<(maximum):
            cnt_with_area.append(c)
            a = cv2.contourArea(c)
            total_area += a
            # total_volume += sqrt(a) ** 3
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
        # print(read_filename)
        out_filename = 'images/result_pics/res_' + os.path.splitext(os.path.basename(f))[0] + '.png'
        result = cv2.drawContours(cv2.imread(read_filename), cnt_with_area, -1, (0, 0, 255), 2)
        cv2.imwrite(out_filename, result)
        ## 畫圖
        # if len(cnt_with_area)!=0.0:
        # print(total_area)
        # print(cnt_with_area)
        avg_area = total_area / len(cnt_with_area)
        # banana_volume = sqrt(avg_area) ** 3
        # avg_volume = total_volume / len(cnt_with_area)
        # banana_volume_list.append(banana_volume)
        # banana_volume_list.append(avg_volume)
        # banana_volume_list.append(total_volume)
        # banana_volume_list.append(total_area)

        buds_area_list.append(avg_area)
    rect()
    # x = list(range(1, len(banana_volume_list) + 1))
    x = np.array([i.toordinal() for i in datetime_objects])
    y_dots = np.array(buds_area_list)

    ###filter array
    # if you don't want data be filtered
    # you can always comment these lines
    for co in range(len(y_dots)):
        if y_dots[co] > 0:
            y_dots = filterData(co, y_dots)
    filter_array = y_dots > 0  # y_dots > 0 equals to return np.where(y_dots!=0)
    x = x[filter_array]
    # print(x)
    y_dots = y_dots[filter_array]  # y_dots[np.where(y_dots != 0)]
    ###filter array end

    ###array of day average
    x_date = []
    y_dots_new = []
    for d in set(x):
        ##put the same date into one element
        x_date.append(d)
        ##calc mean of one day
        y_dots_new.append(np.mean([y_dots[i] for i in range(len(x)) if x[i] == d]))
    ###array end

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

    ax.scatter(x_date, y_dots_new)

    draw_fitline(ax, x_date, y_dots_new, '-', 'exp')

    plt.savefig("scatter.png")

    plt.show()

def filterData(x, y_dots):  # filter unsuitable data and replace with 0
    temp = 0
    c = len(y_dots) - x  # left how much to go
    temp = x
    i = 0;
    while i < c:
        # if y_dots[temp-1] != 0 :
        #     ### filter element with 0
        #     if y_dots[temp] < (y_dots[temp - 1]):
        #         y_dots[temp] = 0
        #         i += 1
        #         temp += 1
        #     else:
        #         temp += 1
        #         i += 1
        # else:
            ### filter element smaller than mean of elements before
        if y_dots[temp] < (np.mean(y_dots[0:temp-1])):
            y_dots[temp] = 0  # element = 0 if y_dots[temp] is lower than mean(y_dots[0:temp-1])
            i += 1
            temp += 1
        else:
            temp += 1
            i += 1
            ###filter end
    return y_dots
