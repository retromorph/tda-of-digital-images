from typings import List

def merge(intervals: List[List[int]]):
    sort(intervals, lambda interval: (interval[0], interval[1]))
    result = []
    currentStart = None
    currentEnd = None
    for i in range(len(intervals) - 1):
        if intervals[i + 1][0] <= intervals[i][1]:
            if currentStart is None:
                currentStart = intervals[i][0]
            currentEnd = max(intervals[i + 1][0], intervals[i + 1][1])
       else:
         if currentStart is not None:
             result.append([currentStart, currentEnd])
             currentStart = None
             currentEnd = None
         else:
           result.append(intervals[i])
     return result