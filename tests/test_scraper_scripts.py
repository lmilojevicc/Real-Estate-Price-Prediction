import unittest

from scraper import collect_urls, run_scraper


class CollectUrlsTests(unittest.TestCase):
    def test_parse_listing_urls_only_reads_offer_cards_and_canonicalizes(self):
        html = """
        <html><body>
          <a href="/stambeni-objekti/stanovi/not-a-card/NkOutside123/">outside</a>
          <div class="advert-list">
            <div class="row offer">
              <h2 class="offer-title">
                <a href="/stambeni-objekti/stanovi/stan-a/NkABC123/?utm=tracking#photos">Stan A</a>
              </h2>
            </div>
            <div class="row offer">
              <h2 class="offer-title">
                <a href="https://www.nekretnine.rs/stambeni-objekti/stanovi/stan-a/NkABC123/">Stan A duplicate</a>
              </h2>
            </div>
            <div class="row offer">
              <h2 class="offer-title">
                <a href="/stambeni-objekti/stanovi/stan-b/NkDEF456/">Stan B</a>
              </h2>
            </div>
          </div>
        </body></html>
        """

        urls = collect_urls.parse_listing_urls(html)

        self.assertEqual(
            urls,
            [
                "https://www.nekretnine.rs/stambeni-objekti/stanovi/stan-a/NkABC123/",
                "https://www.nekretnine.rs/stambeni-objekti/stanovi/stan-b/NkDEF456/",
            ],
        )

    def test_extract_filter_urls_reads_city_data_urls(self):
        html = """
        <div class="filtergroup offer filter">
          <div class="heading">Grad <a class="close closed"></a></div>
          <ul>
            <li><a class="checkbox filter-checkbox" data-url="/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/beograd/lista/po-stranici/20/">Beograd</a></li>
            <li><a class="checkbox filter-checkbox" data-url="/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/novi-sad/lista/po-stranici/20/">Novi Sad</a></li>
          </ul>
        </div>
        <div class="filtergroup offer filter">
          <div class="heading">Grejanje</div>
          <ul><li><a class="checkbox filter-checkbox" data-url="/ignored/">Centralno</a></li></ul>
        </div>
        """

        filters = collect_urls.extract_filter_urls(html, "Grad")

        self.assertEqual(
            filters,
            [
                (
                    "Beograd",
                    "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/beograd/lista/po-stranici/20/",
                ),
                (
                    "Novi Sad",
                    "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/novi-sad/lista/po-stranici/20/",
                ),
            ],
        )

    def test_build_page_url_uses_base_for_first_page(self):
        base_url = "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/beograd/lista/po-stranici/20/"

        self.assertEqual(collect_urls.build_page_url(base_url, 1), base_url)
        self.assertEqual(
            collect_urls.build_page_url(base_url, 2),
            "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/grad/beograd/lista/po-stranici/20/stranica/2/",
        )


class RunScraperTests(unittest.TestCase):
    def test_parse_listing_html_preserves_decimal_area_with_dot_separator(self):
        html = """
        <html><body>
          <h1>Penthouse u BW Residences, 19 i 20 sprat</h1>
          <div class="property__main-details"><ul>
            <li>Kvadratura: 342.85 m²</li>
            <li>Sobe: 5</li>
          </ul></div>
        </body></html>
        """

        row = run_scraper.parse_listing_html(
            html,
            "https://www.nekretnine.rs/stambeni-objekti/stanovi/penthouse-u-bw-residences-19-i-20-sprat/NkxL4LpLwE1/",
        )

        self.assertEqual(row["area_m2"], 342.85)

    def test_parse_listing_html_uses_json_ld_price_and_detail_sections(self):
        html = """
        <html><head>
          <script type="application/ld+json">
          {
            "@context": "http://schema.org",
            "@type": "SoftwareApplication",
            "offers": {"@type": "Offer", "price": "0.00", "priceCurrency": "EUR"}
          }
          </script>
          <script type="application/ld+json">
          {
            "@context": "http://schema.org",
            "@type": "Offer",
            "url": "https://www.nekretnine.rs/stambeni-objekti/stanovi/bulevar-kralja-aleksndra-lux-30-id14424/NknKNVE5630/",
            "priceSpecification": {"price": "199900", "priceCurrency": "EUR"}
          }
          </script>
        </head><body>
          <h1>Bulevar Kralja Aleksndra, lux 3.0 ID#14424</h1>
          <div class="cms-content-inner">Na prodaju lux namešten trosoban stan.</div>
          <div class="property__main-details"><ul>
            <li>Kvadratura: 53 m²</li>
            <li>Sobe: 3</li>
            <li>Grejanje: Centralno</li>
            <li>Parking: Ne</li>
          </ul></div>
          <div class="property__amenities"><ul>
            <li>Spratnost: 5</li>
            <li>Ukupan brој spratova: 8</li>
            <li>Godina izgradnje: 2012</li>
          </ul></div>
          <div class="property__location"><ul>
            <li>Srbija</li>
            <li>Grad Beograd</li>
            <li>Beograd</li>
            <li>Konjarnik</li>
            <li>Bulevar Kralja Aleksandra</li>
          </ul></div>
        </body></html>
        """

        row = run_scraper.parse_listing_html(
            html,
            "https://www.nekretnine.rs/stambeni-objekti/stanovi/bulevar-kralja-aleksndra-lux-30-id14424/NknKNVE5630/",
        )

        self.assertEqual(row["title"], "Bulevar Kralja Aleksndra, lux 3.0 ID#14424")
        self.assertEqual(row["description"], "Na prodaju lux namešten trosoban stan.")
        self.assertEqual(row["price_eur"], 199900.0)
        self.assertEqual(row["area_m2"], 53.0)
        self.assertEqual(row["rooms"], 3.0)
        self.assertEqual(row["city"], "Beograd")
        self.assertEqual(row["region"], "Konjarnik")
        self.assertEqual(row["street"], "Bulevar Kralja Aleksandra")
        self.assertEqual(row["heating_type"], "Centralno")
        self.assertEqual(row["parking"], "Ne")
        self.assertEqual(row["raw_floor_string"], "5 / 8")
        self.assertEqual(row["year_built"], 2012.0)


if __name__ == "__main__":
    unittest.main()
