# -*- coding: utf-8 -*-

"""
Helper used for mailing reports.
"""

import re
from typing import Callable, List, Optional

import argparse as ap
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def parse_source_e_mail(val: str) -> (str, str, str):
    """
    Parse string representation of a source e-mail
    and return its components. The e-mail must be
    in following format:
        ACCOUNT@SERVER:PASSWORD

    :param val: String representation of the source e-mail.

    :raises argparse.ArgumentTypeError: Raised when
        given string is not any of the valid options.

    :return: Returns tuple of (ACCOUNT@SERVER, PASSWORD, SERVER).
    """

    split_specification = re.findall("(.*)@(.*):(.*)", val)
    if len(split_specification) != 1 or len(split_specification[0]) != 3:
        raise ap.ArgumentTypeError(f"Provided source e-mail specification ({val}) is invalid!")

    account = split_specification[0][0]
    server = split_specification[0][1]
    password = split_specification[0][2]

    return f"{account}@{server}", password, server


def parse_e_mail(val: str) -> str:
    """
    Parse string representation of a e-mail and
    return it back. The e-mail must be in following
    format:
        ACCOUNT@SERVER

    :param val: String representation of the e-mail.

    :raises argparse.ArgumentTypeError: Raised when
        given string is not any of the valid options.

    :return: Returns the original string in format ACCOUNT@SERVER.
    """

    split_specification = re.findall("(.*)@(.*)", val)
    if len(split_specification) != 1 or len(split_specification[0]) != 2:
        raise ap.ArgumentTypeError(f"Provided e-mail specification ({val}) is invalid!")

    return val


class Mailer(object):
    """
    Wrapper around mail reporting utilities.
    """

    @classmethod
    def configure(cls, subject_prefix: str, body_footer: str,
                  source_account: str, source_password: str,
                  source_server: str, source_server_port: int,
                  report_list: List[str], runtime_info_fun: Callable,
                  exception_info_fun: Callable):
        """
        Configure the mailer for operation.

        :param subject_prefix: String prefix for each subject.
        :param body_footer: Footer appended to every e-mail body.
        :param source_account: Source e-mail address.
        :param source_password: Source e-mail password.
        :param source_server: Source server to send from.
        :param source_server_port: Port to connect to the source mail
            server.
        :param report_list: List of e-mail addresses (USER@SERVER) to
            send all of the reports to.
        :param runtime_info_fun: Function which returns runtime
            information string.
        :param exception_info_fun: Function which returns exception
            information string.
        """

        cls.subject_prefix = subject_prefix
        cls.body_footer = body_footer

        cls.source_account = source_account
        cls.source_password = source_password
        cls.source_server = source_server
        cls.source_server_port = source_server_port
        cls.target_accounts = report_list

        cls.runtime_info_fun = runtime_info_fun
        cls.exception_info_fun = exception_info_fun

        cls.perform_reporting = True

    @classmethod
    def disable(cls):
        """ Disable e-mail reporting. """

        cls.perform_reporting = False

    def __init__(self):
        pass

    def _connect_to_mail_server(self) -> smtplib.SMTP:
        """
        Connect to the source mail server using configured
        credentials.

        :return: Returns connected SMTP server object.
        """

        # Initialize a secure server connection.
        context = ssl._create_unverified_context();
        server = smtplib.SMTP(Mailer.source_server, Mailer.source_server_port);
        server.connect(Mailer.source_server, Mailer.source_server_port);
        server.starttls(context=context)

        # Login using provided credentials
        server.login(Mailer.source_account, Mailer.source_password)

        return server

    def _finalize_mail_server(self, server: smtplib.SMTP):
        """
        Finalize provided mail server connection and disconnect.
        :param server: Server to finalize.
        """

        server.quit()
        server.close()

    def _generate_message(self, source: str, destination: str,
                          subject: str, body: str) -> str:
        """
        Generate MIME e-mail text from given information.

        :param source: Source e-mail address.
        :param destination: Destination e-mail address.
        :param subject: Subject of the e-mail.
        :param body: Body of the e-mail.

        :return: Returns string representation of the e-mail.
        """

        message = MIMEMultipart("alternative")

        message["From"] = source
        message["To"] = destination
        message["Subject"] = subject

        body_text = MIMEText(body, "plain")
        message.attach(body_text)

        return message.as_string()

    def _generate_message_html(self, source: str, destination: str,
                               subject: str, body_text: str, body_html: str) -> str:
        """
        Generate MIME e-mail text from given information.

        :param source: Source e-mail address.
        :param destination: Destination e-mail address.
        :param subject: Subject of the e-mail.
        :param body_text: Text variant of the e-mail body.
        :param body_html: HTML variant of the e-mail body.

        :return: Returns string representation of the e-mail.
        """

        full_body_html = f"<html><body>\n{body_html}\n</body></html>"

        message = MIMEMultipart()

        message["From"] = source
        message["To"] = destination
        message["Subject"] = subject

        content = MIMEMultipart(
            "alternative",
            None, [MIMEText(body_text), MIMEText(full_body_html, "html")]
        )

        message.attach(content)

        attachment = MIMEText(body_html, _subtype="html")
        attachment.add_header(
            "Content-Disposition", "attachment", filename="body.html")

        message.attach(attachment)

        return message.as_string()

    def report(self, subject: str, body: str):
        """
        Send e-mail with provided subject and body to all
        subscribed report targets.

        :param subject: Subject of the e-mail.
        :param body: Body of the e-mail.
        """

        if not Mailer.perform_reporting:
            return

        try:
            server = self._connect_to_mail_server()
            for destination_account in Mailer.target_accounts:
                message = self._generate_message(
                    source=Mailer.source_account,
                    destination=destination_account,
                    subject=Mailer.subject_prefix + subject,
                    body=body + Mailer.body_footer
                )
                server.sendmail(
                    from_addr=Mailer.source_account,
                    to_addrs=destination_account,
                    msg=message
                )
        except Exception as e:
            print(e)
        else:
            self._finalize_mail_server(server)

    def report_text_html(self, subject: str, body_text: str, body_html: str):
        """
        Send e-mail with provided subject and body to all
        subscribed report targets.

        :param subject: Subject of the e-mail.
        :param body_text: Text variant of the body.
        :param body_html: HTML variant of the body.
        """

        if not Mailer.perform_reporting:
            return

        try:
            server = self._connect_to_mail_server()
            for destination_account in Mailer.target_accounts:
                message = self._generate_message_html(
                    source=Mailer.source_account,
                    destination=destination_account,
                    subject=Mailer.subject_prefix + subject,
                    body_text=body_text + Mailer.body_footer,
                    body_html=body_html + f"<p>{Mailer.body_footer}</p>"
                )
                server.sendmail(
                    from_addr=Mailer.source_account,
                    to_addrs=destination_account,
                    msg=message
                )
        except Exception as e:
            print(e)
        else:
            self._finalize_mail_server(server)

    def report_exception(self, subject: str,
                         exception: Optional[Exception] = None):
        """
        Report on given exception or current exception.

        :param subject: Subject of the e-mail.
        :param exception: Optional exception to report on.
        """

        if not Mailer.perform_reporting:
            return

        text_subject = f"Exception: {subject}"
        text_message = f"Exception occurred during runtime ({subject})!\n\n" \
                       "##############################################################\n" \
                       f"{Mailer.exception_info_fun(exception)}\n" \
                       "##############################################################\n" \
                       f"{Mailer.runtime_info_fun()}"

        self.report(text_subject, text_message)
