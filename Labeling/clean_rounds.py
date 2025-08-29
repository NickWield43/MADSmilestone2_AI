import numpy as np
import pandas as pd


def clean_data(rounds_df):
    # Clean the row values in hc1disescn9, cg1dclkdraw
    # hc1disescn9 asks if subject has dementia or Alzheimer's: 1 YES, 2 NO

    df = rounds_df.copy()  # Create a copy to avoid modifying the input DataFrame
    hc_items = [
        ("2 NO", 2),
        (" 2 NO", 2),
        ("1 YES", 1),
        (" 1 YES", 1),
        ("-9 Missing", np.nan),
        ("-8 DK", np.nan),
        ("7 PREVIOUSLY REPORTED", 7),
        ("-1 Inapplicable", np.nan),
        ("-7 RF", np.nan),
    ]

    cg_items = [
        ("-2 Proxy says cannot ask SP", np.nan),
        ("-7 SP refused to draw clock", np.nan),
        ("-4 SP did not attempt to draw clock", np.nan),
        ("-3 Proxy says can ask SP but SP unable to answer", np.nan),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]

    # Replace values using assignment instead of inplace
    df["hc1disescn9"] = df["hc1disescn9"].replace(
        {item[0]: item[1] for item in hc_items}
    ).astype("float64")  # Use float64 to handle NaN and integers
    df["cg1dclkdraw"] = df["cg1dclkdraw"].replace(
        {item[0]: item[1] for item in cg_items}
    ).astype("float64")  # Use float64 to handle NaN

    # Drop all NaN
    df = df.dropna()

    # Change IDs to string value for streaming images
    df["spid"] = df["spid"].astype("string")

    # Keep just the 8 digit value in spid, removing float value
    df["spid"] = df["spid"].str.extract(r"(\d+).", expand=False)

    return df


def clean_hats_rounds(round_hat_data):
    """Processes the rounds data for all 10 rounds using variables
    that are important for NHATs dementia classification labeling
    """

    df = round_hat_data.copy()  # Create a copy to avoid modifying the input DataFrame

    # Fill NaN values with arbitrary int value
    df["cp1dad8dem"] = df["cp1dad8dem"].fillna(10).astype("int64")

    # Diagnosis variables
    hc_items = [
        ("2 NO", 2),
        (" 2 NO", 2),
        ("1 YES", 1),
        (" 1 YES", 1),
        ("-9 Missing", 10),
        ("-8 DK", 0),
        ("7 PREVIOUSLY REPORTED", 7),
        ("-1 Inapplicable", 10),
        ("-7 RF", 10),
    ]
    ad8dem = [
        ("1 DEMENTIA RESPONSE TO ANY AD8 ITEMS IN PRIOR ROUND", 1),
        ("1 DEMENTIA RESPONSE TO ANY AD8 ITEMS IN PRIOR ROUNDS", 1),
        ("-1 Inapplicable", 0),
    ]

    # Executive Functioning Clock Drawing item
    cg_items = [
        ("-2 Proxy says cannot ask SP", np.nan),
        ("-7 SP refused to draw clock", np.nan),
        ("-4 SP did not attempt to draw clock", np.nan),
        ("-3 Proxy says can ask SP but SP unable to answer", np.nan),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        ("4 Reasonably accurate depiction of a clock", 4),
        ("3 Mildly distorted depiction of a clock", 3),
        ("2 Moderately distorted depection of a clock", 2),
        ("2 Moderately distorted depiction of a clock", 2),
        ("5 Accurate depiction of a clock (circular or square)", 5),
        ("1 Severely distorted depiction of a clock", 1),
        ("0 Not recognizable as a clock", 0),
    ]

    # Orientation Variables
    pres_first = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    pres_last = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    vp_first = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    vp_last = [
        (" 1 Yes", 1),
        ("-1 Inapplicable", 0),
        (" 2 No", 0),
        ("-7 RF", 0),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
    ]
    ans_yr = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_day = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_month = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]
    ans_dow = [
        ("1 YES", 1),
        ("2 NO/DON'T KNOW", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
        (" 1 YES", 1),
        (" 2 NO", 0),
        (" 2 NO/DON'T KNOW", 0),
        ("-7 RF", 0),
    ]

    # Memory Variables
    delay_wrds = [
        ("-3 Proxy says can ask SP but SP unable to answer", 0),
        ("-2 Proxy says cannot ask SP", 0),
        ("-7 SP refused activity", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]
    immed_wrds = [
        ("-3 Proxy says can ask SP but SP unable to answer", 0),
        ("-2 Proxy says cannot ask SP", 0),
        ("-7 SP refused activity", 0),
        ("-1 Inapplicable", np.nan),
        ("-9 Missing", np.nan),
    ]

    # Diagnosis Items
    df["hc1disescn9"] = df["hc1disescn9"].replace(
        {item[0]: item[1] for item in hc_items}
    ).astype("float64")
    df["cp1dad8dem"] = df["cp1dad8dem"].replace(
        {item[0]: item[1] for item in ad8dem}
    ).astype("int64")

    # Executive functioning, clock drawing
    df["cg1dclkdraw"] = df["cg1dclkdraw"].replace(
        {item[0]: item[1] for item in cg_items}
    ).astype("float64")

    # Orientation, Pres Last/First name, VP Last/First, Month, Day, Year, DOW
    df["cg1presidna3"] = df["cg1presidna3"].replace(
        {item[0]: item[1] for item in pres_first}
    ).astype("float64")
    df["cg1presidna1"] = df["cg1presidna1"].replace(
        {item[0]: item[1] for item in pres_last}
    ).astype("float64")
    df["cg1vpname3"] = df["cg1vpname3"].replace(
        {item[0]: item[1] for item in vp_first}
    ).astype("float64")
    df["cg1vpname1"] = df["cg1vpname1"].replace(
        {item[0]: item[1] for item in vp_last}
    ).astype("float64")
    df["cg1todaydat1"] = df["cg1todaydat1"].replace(
        {item[0]: item[1] for item in ans_month}
    ).astype("float64")
    df["cg1todaydat2"] = df["cg1todaydat2"].replace(
        {item[0]: item[1] for item in ans_day}
    ).astype("float64")
    df["cg1todaydat3"] = df["cg1todaydat3"].replace(
        {item[0]: item[1] for item in ans_yr}
    ).astype("float64")
    df["cg1todaydat4"] = df["cg1todaydat4"].replace(
        {item[0]: item[1] for item in ans_dow}
    ).astype("float64")

    # Memory Items, Delay word recall, Immediate Word recall
    df["cg1dwrdimmrc"] = df["cg1dwrdimmrc"].replace(
        {item[0]: item[1] for item in delay_wrds}
    ).astype("float64")
    df["cg1dwrddlyrc"] = df["cg1dwrddlyrc"].replace(
        {item[0]: item[1] for item in immed_wrds}
    ).astype("float64")

    # Change IDs to string value for streaming images
    df["spid"] = df["spid"].astype("string")

    # Keep just the 8 digit value in spid, removing float value
    df["spid"] = df["spid"].str.extract(r"(\d+).", expand=False)

    return df