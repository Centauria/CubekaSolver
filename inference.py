# -*- coding: utf-8 -*-
import argparse
import os.path
from itertools import product, combinations

import numpy as np


def hex2array(code):
    code_array = list(map(int, str(bin(int(code, 16))).lstrip('0b').rjust(68, '0')))
    return code_array


def array2hex(left, top, right, bottom):
    return hex(int(''.join(np.hstack((left, top, right, bottom)).astype(str)), 2)).lstrip('0x').rjust(17, '0')


def split_array(code_array):
    left, top, right, bottom = code_array[:22], code_array[22:34], code_array[34:56], code_array[56:]
    return left, top, right, bottom


def readable(result):
    return ''.join([chr(a + ord('A')) for a in result])


def datify(word):
    return [ord(a) - ord('A') for a in word]


class Inference:
    MASK = -1

    ALL_MASKS = {
        'left': 'fffffc00000000000',
        'top': '000003ffc00000000',
        'right': '000000003fffff000',
        'bottom': '00000000000000fff'
    }

    def __init__(self, data_path):
        self.data_path = data_path
        self.inferences = [np.load(os.path.join(data_path, f'diff_matrix_{i}.npy')) for i in range(4)]

    def search(self, code_array, mask_array=None):
        """Search the database strictly using numeric array."""
        code_array = np.array(code_array)
        if mask_array is not None:
            mask_array = np.array(mask_array)
            code_array[mask_array == 0] = Inference.MASK
        left, top, right, bottom = split_array(code_array)
        result = list(product(range(26), range(26)))
        for i, target in enumerate([left, top, right, bottom]):
            target = np.array(target)
            target_masked = np.where(target != Inference.MASK)[0]
            if target_masked.size > 0:
                result = list(filter(
                    lambda x: (self.inferences[i][x][target_masked] == target[target_masked]).all(), result))
            if len(result) == 0:
                break
        return result

    def search_hex(self, code, mask=None):
        code_array = hex2array(code)
        if mask is not None:
            mask_array = hex2array(mask)
        else:
            mask_array = None
        return self.search(code_array, mask_array)

    def get(self, a, b):
        return np.hstack([inference[a, b] for inference in self.inferences])

    def export_hex(self, a, b):
        return array2hex(*[self.inferences[i][a, b] for i in range(len(self.inferences))])

    def fuzzy_search(self, code, mask=None, max_distance=2, force_fuzzy=False):
        results = []
        min_distance = np.inf
        result = self.search_hex(code, mask)
        if len(result) > 0:
            if not force_fuzzy:
                return result, 0
            results.extend(result)
            min_distance = 0
        code_array = np.array(hex2array(code))
        if mask is not None:
            mask = np.array(hex2array(mask))
        else:
            mask = np.ones_like(code_array)
        for distance in range(1, max_distance + 1):
            print(f'Searching for distance={distance}')
            code_array_masked_part = code_array[mask == 1]
            bits = combinations(range(len(code_array_masked_part)), distance)
            for bit in bits:
                code_altered = np.copy(code_array)
                code_masked_altered = np.copy(code_array_masked_part)
                for b in bit:
                    code_masked_altered[b] = not code_masked_altered[b]
                code_altered[mask == 1] = code_masked_altered
                result = self.search(code_altered, mask)
                if len(result) > 0:
                    results.extend(result)
                    min_distance = min(min_distance, distance)
            if not force_fuzzy and min_distance <= max_distance:
                break
        return results, min_distance

    def infer_sequence(self, codes, mask=None, max_distance=2, force_fuzzy=False):
        if isinstance(mask, str) or mask is None:
            possibilities = [self.fuzzy_search(code, mask, max_distance, force_fuzzy)[0] for code in codes]
        elif isinstance(mask, list):
            assert len(codes) == len(mask)
            possibilities = [self.fuzzy_search(code, m, max_distance, force_fuzzy)[0] for code, m in zip(codes, mask)]
        else:
            raise ValueError('Invalid mask type')

        for i in range(1, len(possibilities)):
            out_linked = {}
            for out in set([x[1] for x in possibilities[i - 1]]):
                linked = False
                for a, b in possibilities[i]:
                    if out == a:
                        linked = True
                out_linked[out] = linked
            for a, b in possibilities[i - 1]:
                if not out_linked[b]:
                    possibilities[i - 1].remove((a, b))
            for a, b in possibilities[i]:
                if a not in set([x[1] for x in possibilities[i - 1]]):
                    possibilities[i].remove((a, b))

        result = [[x[0]] for x in possibilities[0]]

        for i in range(len(possibilities)):
            new_result = []
            for path in result:
                for a, b in possibilities[i]:
                    if path[-1] == a:
                        new_result.append(path + [b])
            result = new_result

        return result

    def encode(self, word):
        data = datify(word)
        data_pair = [data[i:i + 2] for i in range(len(data) - 1)]
        result = [self.export_hex(*pair) for pair in data_pair]
        return result

    def fix(self, a, b, code, mask):
        code_array = np.array(hex2array(code))
        if mask is not None:
            mask_array = np.array(hex2array(mask))
            code_array[mask_array == 0] = Inference.MASK
        left, top, right, bottom = split_array(code_array)
        for i, target in enumerate([left, top, right, bottom]):
            target = np.array(target)
            target_masked = np.where(target != Inference.MASK)[0]
            if target_masked.size > 0:
                self.inferences[i][a, b][target_masked] = target[target_masked]
                self.inferences[i][b, a][target_masked] = target[target_masked]

    def save(self):
        for i, inference in enumerate(self.inferences):
            np.save(os.path.join(self.data_path, f'diff_matrix_{i}.npy'), inference)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument('-d', '--data-folder', default='encoded')
    parser.add_argument('-i', '--input-file', required=True)
    parser.add_argument('--mask-file')
    parser.add_argument('-m', '--mask-kind', choices=['left', 'top', 'right', 'bottom'], required=False)
    parser.add_argument('-f', '--force-fuzzy', action='store_true')
    parser.add_argument('--max-distance', type=int, default=2)
    parser.add_argument('-c', '--calibrate', action='store_true')
    args = parser.parse_args()

    inf = Inference(args.data_folder)
    with open(args.input_file) as f:
        data = list(map(lambda x: x.rstrip(), f.readlines()))
    masks = None
    if args.mask_file is not None:
        with open(args.mask_file) as f:
            masks = list(map(lambda x: x.rstrip(), f.readlines()))
    else:
        if args.mask_kind is not None:
            masks = Inference.ALL_MASKS[args.mask_kind]
    print(data)
    result = inf.infer_sequence(data, masks, args.max_distance, args.force_fuzzy)
    print(len(result))
    if len(result) > 1000:
        result_matrix = np.array(result)
        aggregated = [np.unique(result_matrix[:, i], ) for i in range(result_matrix.shape[1])]
        result_char_list = []
        for item in aggregated:
            if len(item) == 1:
                result_char_list.append(chr(item[0] + ord('A')))
            elif len(item) > 1:
                char_list = map(lambda x: chr(x + ord('A')), item)
                result_char_list.append(f'[{"|".join(char_list)}]')
            else:
                raise ValueError('This is not gonna happen')
        print(''.join(result_char_list))
        aggregated_tuple = [np.unique(result_matrix[:, i:i + 2], axis=0) for i in range(result_matrix.shape[1] - 1)]
        for i, t in enumerate(aggregated_tuple):
            p = len(t)
            print(f'Between index {i} and {i + 1}: {p} possibilities')
            if p < 50:
                s = t + ord('A')
                s = s.astype(np.uint8).view(f'S{s.shape[1]}')
                pos_string = '|'.join([s[i, 0].decode() for i in range(s.shape[0])])
                print(f'({pos_string})')
    else:
        for r in result:
            print(readable(r))
    if args.calibrate:
        correct = input('Please input correct answer(empty line to cancel): ')
        if correct != '':
            assert len(correct) == len(data) + 1
            correct = correct.upper()
            correct_pair = [correct[i:i + 2] for i in range(len(correct) - 1)]
            correction_trace = dict()
            if isinstance(masks, str) or masks is None:
                masks = [masks] * len(data)
            for (wa, wb), x, m in zip(correct_pair, data, masks):
                for a, b in ((wa, wb), (wb, wa)):
                    label = f'{a}{b}'
                    a = ord(a) - ord('A')
                    b = ord(b) - ord('A')
                    original = inf.get(a, b)
                    mask = np.array(hex2array(m))
                    truth = np.array(hex2array(x))
                    if np.any(original[mask == 1] != truth[mask == 1]):
                        response = input(f'"{label}": {inf.export_hex(a, b)} -> {x} [{m}].   Proceed? ([y]/n) :')
                        if response in ('', 'y'):
                            inf.fix(a, b, x, m)
                            after = array2hex(*[inf.inferences[i][a, b] for i in range(len(inf.inferences))])
            response = input('Save? ([y]/n)')
            if response in ('', 'y'):
                inf.save()
