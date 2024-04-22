#TODO: Sort the stations by type - More testing stations?
#Note, there is now a longer list of test stations
ASSETTYPE = [
    "OS",
    "OS",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR"
]

ASSETID = [
    "OS WESTZAANSTRAAT 10-1i",
    "OS MIDDENMEER 20-1i",
    "1 000 798",
    "1 001 859",
    "1 002 623",
    "1 003 729",
    "1 012 959"
]

ASSETTYPE__= [
    "OS",
    "OS",
    "OS",
    "OS",
    "OS",
    "OS",
    "OS",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR",
    "MSR"
]

ASSETID__ = [
    "OS LELYSTAD 10-1i",
    "OS MIDDENMEER 20-1i",
    "OS SNEEK 10-1i",
    "OS WESTZAANSTRAAT 10-1i",
    "OS LEEUWARDEN 10-1i",
    "OS HATTEM 10-1i",
    "OS HEERHUGOWAARD 50-1i",
    "5 002 400",
    "5 000 690",
    "9 011 440",
    "3 004 794",
    "3 016 169",
    "6 002 597",
    "9 001 978",
    "7 011 524",
    "7 003 967",
    "3 023 611"
]

SUMMARY = [
    "Lelystad (OS): veel windmolens",
    "Middenmeer (OS)",
    "Sneek (OS): Standaard",
    "Westzaanstraat (OS)",
    "Leeuwarden (OS)",
    "Hattem (OS)",
    "Heerhugowaard (OS)"
    "MSR 5 002 400: Veel zonnepanelen",
    "MSR 5 000 690: Veel warmtepompen",
    "MSR 9 011 440: Veel warmtepompen (in verhouding)",
    "MSR 3 004 794: Veel warmtepompen (in verhouding)",
    "MSR 3 016 169: weinig WP",
    "MSR 6 002 597: 1/3 WP, weining aansluitingen",
    "MSR 9 001 978: 1/4 WP, weinig aansluitingen",
    "MSR 7 011 524: weinig WP",
    "MSR 7 003 967: veel WP",
    "MSR 3 023 611: veel aansluitingen",
]

Fixed_stations_type = [
    "MSR",
    "MSR",
    "MSR",
    "MSR"
]

Fixed_stations_id = [
"OS MIDDENMEER 20-1i",
"OS WESTZAANSTRAAT 10-1i"
    ]

Fixed_stations_id_ = [
    "0 101 962",
    "0 102 250",
    "0 102 451",
    "0 102 626",
"1 000 361",
"1 000 369",
"1 000 447",
"1 000 465",
"1 000 798",
"1 001 022",
"1 001 199",
"1 001 294",
"1 001 299",
"1 001 309",
"1 001 354",
"1 001 378",
"1 001 410",
"1 001 506",
"1 001 530",
"1 001 859",
"1 002 001",
"1 002 248",
"1 002 293",
"1 002 402",
"1 002 607",
"1 002 623",
"1 002 709",
"1 002 756",
"1 002 981",
"1 003 048",
"1 003 066",
"1 003 092",
"1 003 393",
"1 003 729",
"1 004 195",
"1 004 258",
"1 004 340",
"1 004 453",
"1 004 849",
"1 004 991",
"1 005 100",
"1 005 231",
"1 005 297",
"1 005 470",
"1 005 481",
"1 005 564",
"1 005 651",
"1 005 870",
"1 005 879",
"1 005 949",
"1 006 139",
"1 006 621",
"1 006 738",
"1 006 747",

]

def get_fixed_stations():
    return Fixed_stations_type,Fixed_stations_id
def get_fixed_stations_id():
    return Fixed_stations_id

def get_test_stations():
    return zip(ASSETID,ASSETTYPE)

def get_summary_test_stations():
    print(', '.join(SUMMARY))
    return

