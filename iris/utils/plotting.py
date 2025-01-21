def average_metrics(
        iris_x: list, 
        iris_y: list, 
        resp_x: list, 
        resp_y: list, 
        n: int, 
        m: int
    ) -> tuple[list, list, list, list]:
    '''
    Helper function that averages scores for IRIS and response gene method
    predictions. Specifically for structure of iris.cross_validate_batches.

    Args:
        iris_x: list of x-component of iris predictions (ex. recall)
        iris_y: list of y-component of iris predictions (ex. precision)
        resp_x: list of x-component of response gene method predictions
        resp_y: list of y-component of response gene method predictions
        n: integer number of signals
        m: integer number of batches averaged over

    Returns:
        avgd_iris_x:
        avgd_iris_y:
        avgd_resp_x:
        avgd_resp_y:
    '''
    avgd_iris_y = []
    avgd_iris_x = []
    avgd_resp_y = []
    avgd_resp_x = []

    for i in range(n):
        total_y = 0
        total_x = 0
        for j in range(m):
            total_y += iris_y[j * n + i]
            total_x += iris_x[j * n + i]
        total_y /= m
        total_x /= m 
        avgd_iris_y.append(total_y)
        avgd_iris_x.append(total_x)

        total_y = 0
        total_x = 0
        for j in range(m):
            total_y += resp_y[j * n + i]
            total_x += resp_x[j * n + i]
        total_y /= m
        total_x /= m 
        avgd_resp_y.append(total_y)
        avgd_resp_x.append(total_x)
    
    return avgd_iris_x, avgd_iris_y, avgd_resp_x, avgd_resp_y