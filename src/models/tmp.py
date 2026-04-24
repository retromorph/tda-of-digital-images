from typing import List

def merge(intervals: List[List[int]]):
    intervals.sort()
    result = []
    currentStart = None
    currentEnd = None
    for i in range(len(intervals)):
        if i + 1 == len(intervals):
            result.append([currentStart, currentEnd])
            break
        if currentStart is None:
              currentStart = intervals[i][0]
        if currentEnd is None:
              currentEnd = intervals[i][1]

        if intervals[i + 1][1] <= currentEnd: # включено
            continue
        elif intervals[i + 1][0] <= currentEnd: # пересекается
            currentEnd = intervals[i + 1][1]
        else: # не пересекаемся
             result.append([currentStart, currentEnd])
             currentStart = None
             currentEnd = None
    return result
