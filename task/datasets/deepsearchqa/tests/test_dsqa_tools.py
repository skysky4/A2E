"""DeepSearchQA search ranking and official-URL helpers."""

from ageneval.task.datasets.deepsearchqa.tools import (
    _named_page_search,
    _rank_hits,
    archive_id_url,
    dated_mmddyy,
    order_list_slug,
)


def test_dated_slug_december_15_2014():
    assert dated_mmddyy(
        "During the 2014 Term Year for the United States Supreme Court, "
        "how many certiorari denials were on the Order List on December 15th?"
    ) == "121514"
    assert dated_mmddyy("December 15, 2014 Order List") == "121514"
    assert order_list_slug("site:supremecourt.gov courtorders 121514") == "121514"


def test_rank_hits_drops_fashion_supreme_and_prefers_gov():
    ranked = _rank_hits(
        [
            {"title": "Supreme", "url": "https://supreme.com/", "snippet": ""},
            {
                "title": "dictionary",
                "url": "https://www.merriam-webster.com/dictionary/during",
                "snippet": "",
            },
            {
                "title": "Order List",
                "url": "https://www.supremecourt.gov/orders/courtorders/121514zor_869d.pdf",
                "snippet": "",
            },
        ],
        "United States Supreme Court December 15 2014 Order List certiorari",
    )
    assert ranked
    assert "supremecourt.gov" in ranked[0]["url"]
    assert all("supreme.com" not in h["url"] for h in ranked)
    assert all("merriam-webster" not in h["url"] for h in ranked)
    ranked2 = _rank_hits(
        [{"title": "Orders", "url": "https://www.amazon.com/gp/css/order-history/", "snippet": ""}],
        "Supreme Court December 15 2014 Order List",
    )
    assert ranked2 == []


def test_named_page_search_from_query_wording():
    nhs = _named_page_search("NHS website for Shoulder Pain poor balance")
    assert any("nhs.uk/conditions/shoulder-pain" in h["url"] for h in nhs)
    scotus = _named_page_search(
        "United States Supreme Court Order List on December 15, 2014"
    )
    if scotus:
        assert any("121514" in h["url"] and "supremecourt.gov" in h["url"] for h in scotus)


def test_archive_id_url_wraps_blocked_official_pdf():
    already = "https://web.archive.org/web/20150121000000/https://www.supremecourt.gov/x.pdf"
    stamped = archive_id_url(already)
    assert "20150121000000id_" in stamped
    live = "https://www.supremecourt.gov/orders/courtorders/121514zor_869d.pdf"
    out = archive_id_url(live)
    assert "web.archive.org" in out
    assert "id_" in out
