# -*- coding: utf-8 -*-
"""
Sliding window module

@author: Oumayma Gharbi
"""

#pylint: disable=invalid-name

def slidingWindow(sequence,winSize,step):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence.

    Input sequence must be iterable.

    WinSize : Number of sample points in the window.

    step: Number of points to overlap between segments
    """

    # Verify the inputs
    try:
        it = iter(sequence)  #pylint: disable=unused-variable
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence\
                        length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
    # Do the work
    for i in range(0,int(numOfChunks)*step,step):
        yield sequence[i:i+winSize]
        