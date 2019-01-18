from collections import OrderedDict
import yaml
import os
import sys
import argparse
from time import sleep

from bashutils.colors import color_text


def green(text):
    print(color_text(text, color='green'))

def yellow(text):
    print(color_text(text, color='blue', bcolor='yellow'))

def info(text):
    print(color_text(text, color='white', bcolor='blue'))


class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError, exc:
                raise yaml.constructor.ConstructorError('while constructing a mapping',
                    node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

def main():
    parser = argparse.ArgumentParser(description='Mounts layers')
    parser.add_argument('path', help='Target path')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command to execute')

    my_dir = os.path.realpath('.')

    args = parser.parse_args()

    curdir = os.path.realpath(args.path)

    if curdir == os.getcwd():
        print('Please go out of project directory.')
        sys.exit(1)

    cmd = ' '.join(args.command)

    with open('%s/layers.yml' % curdir) as f:
        config = yaml.load(f, OrderedDictYAMLLoader)

    all_layers = OrderedDict()
    for name, layer in config['layers'].items():
        path = os.path.realpath(os.path.join(curdir, layer['path']))

        if not 'to' in layer:
            to = '/'
        else:
            to = os.path.join(curdir, layer['to'])

        if not os.path.exists(to):
            os.system('mkdir -p %s' % to)

        if not os.path.exists(path):
            if 'create' in layer:
                print('Creating directory %s' % path)
                print(layer['create'])
                cwd = os.getcwd()
                os.chdir(os.path.dirname(path))
                os.system(layer['create'])
                os.chdir(cwd)
            else:
                print('Directory do not exist and no create command provided: %s' % path)
                sys.exit(1)
        if not to in all_layers:
            if to == '/':
                all_layers[to] = [to]
            else:
                all_layers[to] = []
        all_layers[to].append(path)

    # make mount order correct
    if '/' in all_layers:
        root = all_layers['/']
        root[0] = curdir
        del all_layers['/']
        all_layers = OrderedDict([(curdir, root)] + all_layers.items())

    if cmd in ('mount', 'umount'):
        print('Unmounting current layers if mounted ...')

        for to, paths in reversed(all_layers.items()):

            yellow('Unmounting %s' % to)

            command = "grep -qs '%s aufs' /proc/mounts" % to
            is_mounted = os.system(command) == 0

            if is_mounted:
                os.chdir(my_dir)
                command = 'sudo umount %s' % to
                print command
                ret = os.system(command)

                if ret != 0:
                    print('Error happened during umount.')
                    print('\n')
                    sys.exit(1)
                else:
                    green('OK')
            else:
                green('Not mounted.')

    if cmd == 'mount':

        for to, paths in all_layers.items():

            info('Mounting %s' % to)

            os.chdir(my_dir)
            command = 'sudo mount -t aufs -o br=%s -o udba=reval none %s' % (':'.join(['%s' % x for x in all_layers[to]]), to)
            print command
            ret = os.system(command)
            # ret = 0

            if ret != 0:
                print('Error hapend during mount:\n')
                os.system('dmesg | tail -n 1')
                print('\n')
            else:
                green('OK')
                print('Mounted layers:\n')
                print('\n'.join(all_layers[to]) + '\n')

    elif cmd == 'umount':
        pass

    else:

        print('')

        for to, paths in all_layers.items():
            for path in paths:

                info('%s <- %s' % (to, path))
                print('')

                cwd = os.getcwd()
                os.chdir(path)
                os.system(cmd)
                os.chdir(cwd)

                print('')

if __name__ == '__main__':
    main()

