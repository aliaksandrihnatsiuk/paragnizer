def new_split(img, bounds, direction):
    img = img[bounds[1]:bounds[3]+1, bounds[0]:bounds[2]+1]

    x_hist = np.sum(img, axis=0)
    y_hist = np.sum(img, axis=1)

    x = np.where(x_hist != 0)[0]
    y = np.where(y_hist != 0)[0]

    x_bound = np.array((x[0], x[-1]))
    y_bound = np.array((y[0], y[-1]))

    if (direction % 2) == 1:
        rows = np.split(y, np.where(np.diff(y) != 1)[0] + 1)
        bound = x_bound + bounds[0]
        rows = [np.array((row[0], row[-1])) + bounds[1] for row in rows]

    elif (direction % 2) == 0:
        rows = np.split(x, np.where(np.diff(x) != 1)[0] + 1)
        bound = y_bound + bounds[1]
        rows = [np.array((row[0], row[-1])) + bounds[0] for row in rows]

    return rows, bound


def rec_loop(img, bounds, paragraphs, par_id, m=0, direction=1):
    rows, bound = new_split(img, bounds, direction)
    if len(rows) == 1:
        m += 1
    if m == 2:
        if (direction % 2) == 0:
            paragraphs.append([rows[0][0], bound[0], rows[0][1], bound[1]])

        else:
            paragraphs.append([bound[0], rows[0][0], bound[1], rows[0][1]])
        par_id[0] += 1
        return True

    rows_iter = iter(rows)

    while True:
        try:
            row = next(rows_iter)
            if direction % 2:
                bound_ = (bound[0], row[0], bound[-1], row[-1])
            else:
                bound_ = (row[0], bound[0], row[-1], bound[-1])

            rec_loop(img, bound_, paragraphs, par_id, m, direction+1)
        except StopIteration:
            break