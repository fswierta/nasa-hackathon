def dodaj(a, b):
    return a + b


def odejmij(a, b):
    return a - b


def pomnoz(a, b):
    return a * b


def podziel(a, b):
    if b == 0:
        return "Nie można dzielić przez zero"
    return a / b


if __name__ == "__main__":
    a, b = 12, 4
    print(f"{a} + {b} = {dodaj(a, b)}")
    print(f"{a} - {b} = {odejmij(a, b)}")
    print(f"{a} * {b} = {pomnoz(a, b)}")
    print(f"{a} / {b} = {podziel(a, b)}")
