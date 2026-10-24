"""Decode proxy features or compile typed scenes and execute scene queries."""
import argparse
import json

import numpy as np

from .language import execute,parse,render
from .models import ImageFeatureCompiler,ProxyCompiler,SceneCompiler,example


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    for name in ("scene","image-features"):
        command = commands.add_parser(name)
        command.add_argument("--input",help="One two-object .npy feature array; defaults to a bundled example")
        command.add_argument("--query",nargs="+",default=["select","color","red"])
    proxy = commands.add_parser("proxy",help="Predict one of four validated proxy expressions")
    proxy.add_argument("--input",help="One compatible (17,16) .npy proxy array; defaults to a bundled scene example")
    proxy.add_argument("--domain",choices=["scene","robot"],default="scene",
                       help="Input feature hint; does not constrain the predicted expression's domain")
    query = commands.add_parser("query")
    query.add_argument("--program",required=True)
    query.add_argument("--query",nargs="+",required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "proxy":
            values = np.load(args.input,allow_pickle=False) if args.input else example("proxy_example.npy")
            program = ProxyCompiler().compile(values,args.domain)
            result = dict(program=program.render(),fields=dict(operation=program.operation,domain=program.domain,fact=program.fact))
        else:
            if args.command == "query":
                program = parse(args.program)
            else:
                image = args.command == "image-features"
                values = np.load(args.input,allow_pickle=False) if args.input else example("image_example.npy" if image else "scene_example.npy")
                compiler = ImageFeatureCompiler() if image else SceneCompiler()
                program = compiler.compile(values)
            result = dict(program=render(program),query=args.query,
                          answer=execute(program,args.query[0],*args.query[1:]))
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
