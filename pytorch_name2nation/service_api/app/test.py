from name2lang import NameToNationClassify

def main():
    model = NameToNationClassify("../models")
    model.load_state()
    rc = model.predict("flash")
    print(f"{rc=}")

if __name__ == '__main__':
    main()