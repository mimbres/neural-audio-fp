# -*- coding: utf-8 -*-
""" print_table.py """
import curses
import numpy as np


class PrintTable():
    def __init__(self, scr, test_seq_len, row_names):
        self.color_cyan = '\033[36m'
        self.color_def = '\033[0m'
        self.line_int = '{:^6}\t' * len(test_seq_len) # columns
        self.line_float = '{:>4.2f}\t' * len(test_seq_len)

        # Init curses window
        self.scr = scr #curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)

        # title
        self.scr.addstr(
            0, 0,
            '========== Top-N hit rate (%) of segment-level search ==========')

        # column title
        self.scr.addstr(2, 14, '{:^43}\t'.format(
            '------------------- Query length ------------------'))

        # column names
        self.scr.addstr(
            3, 14, self.line_int.format(*test_seq_len), curses.color_pair(1))
        test_seq_len_sec = (np.asarray(test_seq_len) + 1) // 2
        test_seq_len_sec = ['(' + str(i) + 's)' for i in test_seq_len_sec]
        self.scr.addstr(
            4, 14, self.line_int.format(*test_seq_len_sec),
            curses.color_pair(1))
        self.test_seq_len = test_seq_len
        self.test_seq_len_sec = test_seq_len_sec

        # row names
        self.scr.addstr(3, 0,  '{:^14}'.format('segments'))
        self.scr.addstr(4, 0,  '{:^14}'.format('seconds'))
        for i, r in enumerate(row_names):
            self.scr.addstr(i + 6, 0,  '{:^14}'.format(r))
        self.row_names = row_names

        self.scr.refresh()
        self.rows_cache = None
        self.avg_search_time_cache = None


    def update_table(self, rows):
        """

        Parameters
        ----------
        row_values : list([x,...], [x,..],...)

        """
        self.rows_cache = rows
        for i, line in enumerate(rows):
            self.scr.addstr(i + 6, 14, self.line_float.format(*line))
        self.scr.refresh()


    def update_counter(self, i, niter, t):
        """

        i: current count
        niter: number or iterations
        t: time in ms

        """
        self.scr.addstr(10, 2, f'{i}/{niter}', curses.color_pair(2))
        self.scr.addstr(11, 2, f'{t:>4.2f} ms/query', curses.color_pair(2))
        self.avg_search_time_cache = t

    def close_table(self):
        curses.endwin()
        print('========= Top1 hit rate (%) of segment-level search =========')
        print(' ' * 14, '{:^43}\t'.format('---------------- Query length ----------------') )
        print('{:^14}'.format('segments'), self.color_cyan + self.line_int.format(*(self.test_seq_len)), self.color_def)
        print('{:^14}'.format('seconds'), self.color_cyan + self.line_int.format(*(self.test_seq_len_sec)), self.color_def)
        print('')
        for i, line in enumerate(self.rows_cache):
            print('{:^14}'.format(self.row_names[i]),
                  self.line_float.format(*line))
        print('=============================================================')
        print(f'average search + evaluation time {self.avg_search_time_cache:>4.2f} ms/query')


def test():
    import time
    pt = PrintTable(
        test_seq_len=[1, 3, 5, 9, 11, 19],
        row_names=['Top1 exact', 'Top1 near', 'Confidence'])

    niter = 15
    for i in range(niter):
        t = (i + 1) * 1000.
        pt.update_counter(i, niter, t)
        values = (np.random.rand(6), np.random.rand(6), np.random.rand(6))
        pt.update_table(values)
        time.sleep(0.5)
    pt.close_table()


if __name__ == "__main__":
    test()
