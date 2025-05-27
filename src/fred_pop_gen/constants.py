from enum import Enum


class Enrollment(Enum):
    """Represents person school enrollment status."""

    PUBLIC = 0
    PRIVATE = 1
    NOT_ENROLLED = 2


class Grade(Enum):
    """Represents school grade levels."""

    PREK = 0
    K = 1
    FIRST = 2
    SECOND = 3
    THIRD = 4
    FOURTH = 5
    FIFTH = 6
    SIXTH = 7
    SEVENTH = 8
    EIGHTH = 9
    NINTH = 10
    TENTH = 11
    ELEVENTH = 12
    TWELFTH = 13

    @staticmethod
    def range(lower_bound: "Grade", upper_bound: "Grade") -> set["Grade"]:
        return {
            Grade(value) for value in range(lower_bound.value, upper_bound.value + 1)
        }


class EmploymentAgeBucket(Enum):
    """Represents employment age buckets"""

    B_UNDER_16 = 0
    B_16_TO_19 = 1
    B_20_TO_21 = 2
    B_22_TO_24 = 3
    B_25_TO_29 = 4
    B_30_TO_34 = 5
    B_35_TO_44 = 6
    B_45_TO_54 = 7
    B_55_TO_59 = 8
    B_60_TO_61 = 9
    B_62_TO_64 = 10
    B_65_TO_69 = 11
    B_70_TO_74 = 12
    B_75_PLUS = 13

    @staticmethod
    def get_bucket(age: int) -> "EmploymentAgeBucket":
        if age < 16:
            return EmploymentAgeBucket.B_UNDER_16
        elif age < 19:
            return EmploymentAgeBucket.B_16_TO_19
        elif 16 <= age <= 19:
            return EmploymentAgeBucket.B_16_TO_19
        elif 20 <= age <= 21:
            return EmploymentAgeBucket.B_20_TO_21
        elif 22 <= age <= 24:
            return EmploymentAgeBucket.B_22_TO_24
        elif 25 <= age <= 29:
            return EmploymentAgeBucket.B_25_TO_29
        elif 30 <= age <= 34:
            return EmploymentAgeBucket.B_30_TO_34
        elif 35 <= age <= 44:
            return EmploymentAgeBucket.B_35_TO_44
        elif 45 <= age <= 54:
            return EmploymentAgeBucket.B_45_TO_54
        elif 55 <= age <= 59:
            return EmploymentAgeBucket.B_55_TO_59
        elif 60 <= age <= 61:
            return EmploymentAgeBucket.B_60_TO_61
        elif 62 <= age <= 64:
            return EmploymentAgeBucket.B_62_TO_64
        elif 65 <= age <= 69:
            return EmploymentAgeBucket.B_65_TO_69
        elif 70 <= age <= 74:
            return EmploymentAgeBucket.B_70_TO_74
        else:
            return EmploymentAgeBucket.B_75_PLUS
